use std::{
    fs::File,
    io::{BufWriter, Cursor, Write},
    sync::{Arc, Mutex},
};

use cpal::{
    FromSample, Sample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use hound::WavWriter;
use opus::{Application, Channels, Encoder};

struct OpusFileWriter {
    file: File,
    serial: u32,
    page_sequence: u32,
    granule_position: u64,
    frame_size: usize,
}

impl OpusFileWriter {
    fn new(path: &str, sample_rate: u32, channels: u8) -> std::io::Result<Self> {
        let mut writer = OpusFileWriter {
            file: File::create(path)?,
            serial: rand::random(),
            page_sequence: 0,
            granule_position: 0,
            frame_size: match sample_rate {
                48000 => 960,
                24000 => 480,
                16000 => 320,
                _ => 960,
            },
        };

        // Write Opus identification header
        writer.write_id_header(sample_rate, channels)?;

        // Write Opus comment header
        writer.write_comment_header()?;

        Ok(writer)
    }

    fn write_id_header(&mut self, sample_rate: u32, channels: u8) -> std::io::Result<()> {
        let mut header = Vec::new();
        header.extend_from_slice(b"OpusHead");
        header.push(1); // Version
        header.push(channels);
        header.extend_from_slice(&3840u16.to_le_bytes()); // Pre-skip (80ms at 48kHz)
        header.extend_from_slice(&sample_rate.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes()); // Output gain
        header.push(0); // Channel mapping family

        self.write_ogg_page(&header, 0, 0x02)?; // BOS flag
        Ok(())
    }

    fn write_comment_header(&mut self) -> std::io::Result<()> {
        let vendor = b"Rust Opus Encoder";
        let mut header = Vec::new();
        header.extend_from_slice(b"OpusTags");
        header.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        header.extend_from_slice(vendor);
        header.extend_from_slice(&0u32.to_le_bytes()); // No user comments

        self.write_ogg_page(&header, 0, 0)?;
        Ok(())
    }

    fn write_audio_packet(&mut self, packet: &[u8]) -> std::io::Result<()> {
        self.granule_position += self.frame_size as u64;
        self.write_ogg_page(packet, self.granule_position, 0)?;
        Ok(())
    }

    fn finalize(&mut self) -> std::io::Result<()> {
        // Write empty page with EOS flag
        self.write_ogg_page(&[], self.granule_position, 0x04)?;
        Ok(())
    }

    fn write_ogg_page(&mut self, data: &[u8], granule: u64, flags: u8) -> std::io::Result<()> {
        let num_segments = ((data.len() + 254) / 255).max(1);
        let mut segments = vec![255u8; num_segments];

        if !data.is_empty() {
            segments[num_segments - 1] = (data.len() % 255) as u8;
            if segments[num_segments - 1] == 0 {
                segments[num_segments - 1] = 255;
            }
        } else {
            segments[0] = 0;
        }

        let mut page = Vec::new();
        page.extend_from_slice(b"OggS"); // Capture pattern
        page.push(0); // Version
        page.push(flags); // Header type
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&self.serial.to_le_bytes());
        page.extend_from_slice(&self.page_sequence.to_le_bytes());
        page.extend_from_slice(&[0u8; 4]); // CRC (placeholder)
        page.push(segments.len() as u8);
        page.extend_from_slice(&segments);
        page.extend_from_slice(data);

        // Calculate CRC32
        let crc = Self::crc32(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());

        self.file.write_all(&page)?;
        self.page_sequence += 1;
        Ok(())
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0u32;
        for &byte in data {
            crc ^= (byte as u32) << 24;
            for _ in 0..8 {
                if crc & 0x80000000 != 0 {
                    crc = (crc << 1) ^ 0x04c11db7;
                } else {
                    crc <<= 1;
                }
            }
        }
        crc
    }
}

fn encode_to_opus_file(
    pcm_data: Vec<f32>,
    sample_rate: u32,
    channels: Channels,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create encoder
    let mut encoder = Encoder::new(sample_rate, channels, Application::Audio)?;

    // Convert f32 to i16
    let pcm_i16: Vec<i16> = pcm_data
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();

    let frame_size = match sample_rate {
        48000 => 960,
        24000 => 480,
        16000 => 320,
        _ => 960,
    };

    let ch_count = match channels {
        Channels::Mono => 1,
        Channels::Stereo => 2,
    };

    let samples_per_frame = frame_size * ch_count;
    let mut writer = OpusFileWriter::new(output_path, sample_rate, ch_count as u8)?;
    let mut encoded_frame = vec![0u8; 4000];

    // Encode and write frames
    for chunk in pcm_i16.chunks(samples_per_frame) {
        if chunk.len() == samples_per_frame {
            let len = encoder.encode(chunk, &mut encoded_frame)?;
            writer.write_audio_packet(&encoded_frame[..len])?;
        }
    }

    writer.finalize()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let audio_data = Arc::new(Mutex::new(Vec::<f32>::new()));
    let audio_data_c = audio_data.clone();

    let host = cpal::default_host();

    // Get the default *output* device (speaker)
    let device = host
        .default_output_device()
        .expect("no output device available");

    // On Windows, output devices can be used in loopback mode to capture their playback.
    let config = device.default_output_config()?.config();

    println!("{:?}", config);

    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/recorded.wav");
    println!("PATH {}", PATH);
    let spec = wav_spec_from_config(&device.default_output_config().unwrap());
    let writer = hound::WavWriter::create(PATH, spec)?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    let writer_2 = writer.clone();

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            // Encode float samples to Opus

            write_input_data::<f32, f32>(data, &writer_2);

            audio_data.lock().unwrap().extend_from_slice(data);
        },
        |err| eprintln!("Stream error: {err}"),
        None,
    )?;

    stream.play()?;
    std::thread::sleep(std::time::Duration::from_secs(10));

    match encode_to_opus_file(
        audio_data_c.lock().unwrap().to_vec(),
        config.sample_rate.0 as u32,
        match config.channels {
            1 => Channels::Mono,
            _ => Channels::Stereo,
        },
        "output.opus",
    ) {
        Ok(_) => println!("✓ Successfully created playable output.opus"),
        Err(e) => eprintln!("Error: {}", e),
    }

    let pcm_data: Vec<u8> =
        write_audio_to_memory(&audio_data_c.lock().unwrap().to_vec(), 16000).unwrap();

    let pcm_data = pcm_u8_to_f32(&pcm_data);

    match encode_to_opus_file(
        pcm_data,
        config.sample_rate.0 as u32,
        match config.channels {
            1 => Channels::Mono,
            _ => Channels::Stereo,
        },
        "output.opus",
    ) {
        Ok(_) => println!("✓ Successfully created playable output.opus"),
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}

fn pcm_u8_to_f32(pcm_bytes: &[u8]) -> Vec<f32> {
    use std::convert::TryInto;

    pcm_bytes
        .chunks_exact(2)
        .map(|b| {
            let sample = i16::from_le_bytes(b.try_into().unwrap());
            sample as f32 / i16::MAX as f32
        })
        .collect()
}

fn wav_spec_from_config(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate().0 as _,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: sample_format(config.sample_format()),
    }
}

fn sample_format(format: cpal::SampleFormat) -> hound::SampleFormat {
    if format.is_float() {
        hound::SampleFormat::Float
    } else {
        hound::SampleFormat::Int
    }
}

type WavWriterHandle = Arc<Mutex<Option<hound::WavWriter<BufWriter<File>>>>>;

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
where
    T: Sample,
    U: Sample + hound::Sample + FromSample<T>,
{
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input.iter() {
                let sample: U = U::from_sample(sample);
                writer.write_sample(sample).ok();
            }
        }
    }
}

fn write_audio_to_memory(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer: WavWriter<&mut Cursor<Vec<u8>>> = hound::WavWriter::new(&mut cursor, spec)?;

    for sample in samples {
        writer.write_sample((*sample * i16::MAX as f32) as i16)?;
    }

    writer.finalize()?;

    let wav_data = cursor.into_inner();

    Ok(wav_data)
}
