//! Minimal ICMPv6 echo prober for IPv6 TGA evaluation.
//!
//! Architecture
//! ------------
//! Two threads share ownership of the scan state:
//!
//!   sender   — reads addresses from file, sends ICMPv6 Echo Requests at the
//!              configured rate (default 20 kpps), inserts each (seq → addr)
//!              into the pending map.
//!
//!   receiver — continuously drains the ICMPv6 socket; for every Echo Reply
//!              whose identifier matches this session, looks up the seq in the
//!              pending map and records the address as a hit.
//!
//! Rate limiting
//! -------------
//! Packet sends are batched per millisecond:
//!   packets_per_ms = rate_pps / 1000
//! After each batch the sender sleeps for the remainder of that millisecond.
//! This keeps average rate accurate without per-packet spin-waiting.
//!
//! Sequence numbers
//! ----------------
//! A 16-bit seq counter wraps every 65 535 packets (~3.3 s at 20 kpps).
//! The 3-second reply timeout means in-flight entries are almost always gone
//! before seq reuses a slot, so collision probability is negligible.
//!
//! Usage
//! -----
//!   sudo ./scanner --input candidates.txt --output hits.txt
//!   sudo ./scanner --input candidates.txt --output hits.txt --rate 10000 --timeout 5
//!
//! Requires root or CAP_NET_RAW on Linux.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::{IpAddr, Ipv6Addr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use dashmap::DashMap;
use pnet::packet::icmpv6::echo_request::{EchoRequestPacket, MutableEchoRequestPacket};
use pnet::packet::icmpv6::{Icmpv6Code, Icmpv6Types, MutableIcmpv6Packet};
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::Packet;
use pnet::transport::{icmpv6_packet_iter, transport_channel, TransportChannelType,
                      TransportProtocol};
use rand::random;

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name    = "scanner",
    about   = "Minimal ICMPv6 echo prober for IPv6 TGA evaluation",
    version
)]
struct Args {
    /// Input file: one IPv6 address per line
    #[arg(short, long)]
    input: String,

    /// Output file: responsive (hit) IPv6 addresses, one per line
    #[arg(short, long)]
    output: String,

    /// Scanning rate in packets per second
    #[arg(short, long, default_value = "20000")]
    rate: u64,

    /// Seconds to wait after last packet sent before stopping
    #[arg(short, long, default_value = "3")]
    timeout: u64,

    /// Log progress to stderr every N packets (0 = disable)
    #[arg(long, default_value = "100000")]
    progress: u64,
}

// ─── Pending-map entry ───────────────────────────────────────────────────────

struct Entry {
    addr: String,
    #[allow(dead_code)]
    sent_at: Instant,
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // Random identifier distinguishes our packets from any other ICMPv6 traffic.
    let identifier: u16 = random();

    // ── Shared state ────────────────────────────────────────────────────────
    let pending: Arc<DashMap<u16, Entry>> = Arc::new(DashMap::new());
    let hits:    Arc<Mutex<Vec<String>>>  = Arc::new(Mutex::new(Vec::new()));

    let done_sending: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let sent_count:   Arc<AtomicU64>  = Arc::new(AtomicU64::new(0));
    let hit_count:    Arc<AtomicU64>  = Arc::new(AtomicU64::new(0));

    // ── Open ICMPv6 transport channel (needs root / CAP_NET_RAW) ────────────
    let protocol = TransportChannelType::Layer4(
        TransportProtocol::Ipv6(IpNextHeaderProtocols::Icmpv6),
    );
    let (tx, rx) = transport_channel(1 << 20, protocol)
        .context("Failed to open ICMPv6 transport channel — need root or CAP_NET_RAW")?;

    // ── Spawn receiver thread ────────────────────────────────────────────────
    let pending_r   = Arc::clone(&pending);
    let hits_r      = Arc::clone(&hits);
    let done_r      = Arc::clone(&done_sending);
    let hit_count_r = Arc::clone(&hit_count);
    let timeout_sec = args.timeout;

    let recv_handle = thread::spawn(move || {
        let mut rx = rx;
        let mut iter = icmpv6_packet_iter(&mut rx);

        // Once sender signals done, we give replies `timeout_sec` more seconds.
        let mut deadline: Option<Instant> = None;

        loop {
            // Set deadline the first time we see done_sending == true.
            if deadline.is_none() && done_r.load(Ordering::Relaxed) {
                deadline = Some(Instant::now() + Duration::from_secs(timeout_sec));
            }

            // Stop when deadline has passed.
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    break;
                }
            }

            // Poll with a short timeout so we can re-check the deadline.
            match iter.next_with_timeout(Duration::from_millis(100)) {
                Ok(Some((icmpv6_pkt, _src))) => {
                    // Only care about Echo Replies (type 129).
                    if icmpv6_pkt.get_icmpv6_type() != Icmpv6Types::EchoReply {
                        continue;
                    }

                    // Echo Reply has the same binary layout as Echo Request.
                    if let Some(echo) = EchoRequestPacket::new(icmpv6_pkt.packet()) {
                        if echo.get_identifier() != identifier {
                            continue; // Not from our session.
                        }
                        let seq = echo.get_sequence_number();
                        if let Some((_, entry)) = pending_r.remove(&seq) {
                            hit_count_r.fetch_add(1, Ordering::Relaxed);
                            hits_r.lock().unwrap().push(entry.addr);
                        }
                    }
                }
                Ok(None) => {} // Timeout on this poll — loop again.
                Err(e) => {
                    eprintln!("[scanner] receiver error: {e}");
                    break;
                }
            }
        }
    });

    // ── Sender loop (main thread) ────────────────────────────────────────────
    let in_file = File::open(&args.input)
        .with_context(|| format!("Cannot open input file: {}", args.input))?;
    let reader = BufReader::new(in_file);

    // Batch-based rate limiting: sleep at the end of each 1 ms window.
    let rate_pps       = args.rate.max(1);
    let packets_per_ms = (rate_pps / 1000).max(1);

    let mut tx      = tx;
    let mut seq: u16 = 0;
    let start       = Instant::now();

    // Batch window
    let mut batch_count: u64 = 0;
    let mut batch_start      = Instant::now();

    for line in reader.lines() {
        let line     = line?;
        let addr_str = line.trim().to_string();
        if addr_str.is_empty() || addr_str.starts_with('#') {
            continue;
        }

        let addr: Ipv6Addr = match addr_str.parse() {
            Ok(a)  => a,
            Err(_) => {
                eprintln!("[scanner] skipping invalid address: {addr_str}");
                continue;
            }
        };

        // ── Rate limiting ───────────────────────────────────────────────────
        batch_count += 1;
        if batch_count >= packets_per_ms {
            let elapsed = batch_start.elapsed();
            if elapsed < Duration::from_millis(1) {
                thread::sleep(Duration::from_millis(1) - elapsed);
            }
            batch_start = Instant::now();
            batch_count = 0;
        }

        // ── Build ICMPv6 Echo Request ───────────────────────────────────────
        // Buffer: 8 bytes ICMPv6 header + 8 bytes zero payload = 16 bytes.
        let mut buf = [0u8; 16];
        {
            let mut pkt = MutableEchoRequestPacket::new(&mut buf)
                .expect("buffer too small");
            pkt.set_icmpv6_type(Icmpv6Types::EchoRequest);
            pkt.set_icmpv6_code(Icmpv6Code::new(0));
            pkt.set_identifier(identifier);
            pkt.set_sequence_number(seq);
            // Checksum is computed by the kernel for raw ICMPv6 sockets.
        }

        // Insert into pending *before* sending to avoid a race where the
        // reply arrives before we record the entry.
        pending.insert(seq, Entry { addr: addr_str, sent_at: Instant::now() });

        let send_pkt = MutableEchoRequestPacket::new(&mut buf)
            .expect("buffer too small");

        if let Err(e) = tx.send_to(send_pkt, IpAddr::V6(addr)) {
            eprintln!("[scanner] send error for {addr}: {e}");
            pending.remove(&seq);
        }

        seq = seq.wrapping_add(1);
        let n = sent_count.fetch_add(1, Ordering::Relaxed) + 1;

        // ── Progress reporting ──────────────────────────────────────────────
        if args.progress > 0 && n % args.progress == 0 {
            let hits = hit_count.load(Ordering::Relaxed);
            let secs = start.elapsed().as_secs_f64();
            let pps  = n as f64 / secs;
            eprintln!(
                "[scanner] sent={n:>10}  hits={hits:>8}  rate={pps:>8.0} pps  \
                 elapsed={secs:.1}s"
            );
        }
    }

    // ── Signal receiver and wait ─────────────────────────────────────────────
    let total_sent = sent_count.load(Ordering::Relaxed);
    eprintln!(
        "[scanner] Sending complete ({total_sent} packets). \
         Waiting {timeout_sec}s for remaining replies …"
    );
    done_sending.store(true, Ordering::Relaxed);
    recv_handle.join().expect("receiver thread panicked");

    // ── Write output ─────────────────────────────────────────────────────────
    let total_hits = hit_count.load(Ordering::Relaxed);
    let hit_rate   = if total_sent > 0 {
        total_hits as f64 / total_sent as f64 * 100.0
    } else {
        0.0
    };

    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&args.output)
        .with_context(|| format!("Cannot open output file: {}", args.output))?;
    let mut writer = BufWriter::new(out_file);

    for addr in hits.lock().unwrap().iter() {
        writeln!(writer, "{addr}")?;
    }

    eprintln!(
        "[scanner] Done — sent={total_sent}  hits={total_hits}  \
         hit_rate={hit_rate:.2}%  output={}",
        args.output
    );

    Ok(())
}
