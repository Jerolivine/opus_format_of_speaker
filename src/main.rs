use opus::{Application, Channels, Encoder};

fn encode_to_opus(
    pcm_data: Vec<f32>,
    sample_rate: u32,
    channels: Channels,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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
    let mut encoded_frame = vec![0u8; 4000];

    // Encode and write frames
    for chunk in pcm_i16.chunks(samples_per_frame) {
        if chunk.len() == samples_per_frame {
            encoder.encode(chunk, &mut encoded_frame)?;
        }
    }

    Ok(encoded_frame)
}

fn main() -> anyhow::Result<()> {
    match encode_to_opus(Vec::new(), 16000, Channels::Stereo) {
        Ok(_) => println!("✓ Successfully created playable output.opus"),
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}
