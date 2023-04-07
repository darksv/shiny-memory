extern crate ffmpeg_next as ffmpeg;

use std::path::Path;

use anyhow::Context as _;
use ffmpeg::{codec, decoder, encoder, ffi, format, frame, log, Packet, StreamMut};
use ffmpeg::format::context::Output;
use ffmpeg::format::Pixel;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use image::{ColorType, Rgb};
use image::flat::SampleLayout;

pub(crate) fn init() {
    ffmpeg::init().unwrap();
    log::set_level(log::Level::Info);
}

fn clear_codec_tag(stream: &mut StreamMut) {
    // We need to set codec_tag to 0 lest we run into incompatible codec tag
    // issues when muxing into a different container format. Unfortunately
    // there's no high level API to do this (yet).
    unsafe {
        (*stream.parameters().as_mut_ptr()).codec_tag = 0;
    }
}

pub(crate) fn overlay_video<FrameCallback>(
    input_file: impl AsRef<Path>,
    output_file: impl AsRef<Path>,
    mut callback: FrameCallback,
) -> anyhow::Result<()>
    where
        FrameCallback: FnMut(usize, &mut image::flat::ViewMut<&mut [u8], Rgb<u8>>)
{
    let mut ictx = format::input(&input_file)?;
    let mut octx = format::output(&output_file)?;

    let input_video_stream = ictx.streams().best(Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?;
    let input_video_stream_index = input_video_stream.index();
    let video_time_base = input_video_stream.time_base();

    let input_audio_stream = ictx.streams().best(Type::Audio);
    let input_audio_stream_index = input_audio_stream.as_ref().map(|s| s.index());
    let audio_time_base = input_audio_stream.as_ref().map(|it| it.time_base());

    let mut output_audio_index = None;

    let mut ost = octx.add_stream(encoder::find(codec::Id::H264))?;
    clear_codec_tag(&mut ost);
    let output_video_index = ost.index();

    if let Some(audio) = input_audio_stream {
        let mut ost = octx.add_stream(encoder::find(codec::Id::None))?;
        ost.set_parameters(audio.parameters());
        clear_codec_tag(&mut ost);
        output_audio_index = Some(ost.index());
    }

    let context_decoder = codec::context::Context::from_parameters(input_video_stream.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;

    let mut input_scaler = Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        Flags::BILINEAR,
    )?;

    let mut output_video_stream = octx.stream_mut(output_video_index).unwrap();

    let codec = encoder::find(codec::Id::H264).unwrap();
    let context_encoder = unsafe { codec::context::Context::wrap(ffi::avcodec_alloc_context3(codec.as_ptr()), None) };
    let mut encoder = context_encoder
        .encoder()
        .video()?;

    encoder.set_height(decoder.height());
    encoder.set_width(decoder.width());
    encoder.set_aspect_ratio(decoder.aspect_ratio());
    encoder.set_format(decoder.format());
    encoder.set_frame_rate(decoder.frame_rate());
    encoder.set_time_base(input_video_stream.time_base());
    output_video_stream.set_parameters(&encoder);

    let mut encoder = encoder.open().unwrap();
    let mut output_scaler = Context::get(
        Pixel::RGB24,
        encoder.width(),
        encoder.height(),
        encoder.format(),
        encoder.width(),
        encoder.height(),
        Flags::BILINEAR,
    )?;

    let mut frame_index = 0;
    let mut process_frames =
        |decoder: &mut decoder::Video, encoder: &mut encoder::video::Video| -> anyhow::Result<()> {
            let mut decoded = frame::Video::empty();
            let mut encoded = frame::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut frame = frame::Video::empty();
                input_scaler.run(&decoded, &mut frame)?;
                encoded.set_pts(decoded.pts());

                let mut flat_samples = image::FlatSamples {
                    layout: sample_layout_of_frame(&frame),
                    color_hint: Some(ColorType::Rgb8),
                    samples: frame.data_mut(0),
                };
                let mut view = flat_samples.as_view_mut()
                    .context("creating view")?;
                callback(frame_index, &mut view);

                output_scaler.run(&frame, &mut encoded)?;
                encoder.send_frame(&encoded)?;

                frame_index += 1;
            }
            Ok(())
        };

    octx.set_metadata(ictx.metadata().to_owned());
    octx.write_header()?;

    let encoded_into_packets = |encoder: &mut encoder::Encoder, output: &mut Output| {
        let mut packet = Packet::empty();
        while encoder.receive_packet(&mut packet).is_ok() {
            let ost = output.stream(output_video_index).unwrap();
            packet.rescale_ts(video_time_base, ost.time_base());
            packet.set_position(-1);
            packet.set_stream(output_video_index);
            packet.write_interleaved(output).unwrap();
        }
    };

    for (stream, mut packet) in ictx.packets() {
        match stream.parameters().medium() {
            Type::Video if input_video_stream_index == stream.index() => {
                decoder.send_packet(&packet)?;
                process_frames(&mut decoder, &mut encoder)?;
                encoded_into_packets(&mut encoder, &mut octx);
            }
            Type::Audio if input_audio_stream_index == Some(stream.index()) => {
                let ost = octx.stream(output_audio_index.unwrap()).unwrap();
                packet.rescale_ts(audio_time_base.unwrap(), ost.time_base());
                packet.set_position(-1);
                packet.set_stream(output_audio_index.unwrap());
                packet.write_interleaved(&mut octx)?;
            }
            _ => (),
        }
    }

    decoder.send_eof()?;
    process_frames(&mut decoder, &mut encoder)?;
    encoded_into_packets(&mut encoder, &mut octx);
    encoder.send_eof()?;
    encoded_into_packets(&mut encoder, &mut octx);
    octx.write_trailer()?;

    Ok(())
}

pub(crate) fn decode_video<FrameCallback>(
    input_file: impl AsRef<Path>,
    mut callback: FrameCallback,
) -> anyhow::Result<()>
    where
        FrameCallback: FnMut(usize, &image::flat::View<&[u8], Rgb<u8>>)
{
    let mut ictx = format::input(&input_file)?;

    let input_video_stream = ictx.streams().best(Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?;
    let input_video_stream_index = input_video_stream.index();

    let context_decoder = codec::context::Context::from_parameters(input_video_stream.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;

    let mut input_scaler = Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        Flags::BILINEAR,
    )?;

    let mut frame_index = 0;
    let mut process_frames =
        |decoder: &mut decoder::Video| -> anyhow::Result<()> {
            let mut decoded = frame::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut frame = frame::Video::empty();
                input_scaler.run(&decoded, &mut frame)?;

                let flat_samples = image::FlatSamples {
                    layout: sample_layout_of_frame(&frame),
                    color_hint: Some(ColorType::Rgb8),
                    samples: frame.data(0),
                };
                let view = flat_samples.as_view()
                    .context("creating view")?;
                callback(frame_index, &view);

                frame_index += 1;
            }
            Ok(())
        };

    for (stream, packet) in ictx.packets() {
        match stream.parameters().medium() {
            Type::Video if input_video_stream_index == stream.index() => {
                decoder.send_packet(&packet)?;
                process_frames(&mut decoder)?;
            }
            _ => (),
        }
    }

    decoder.send_eof()?;
    process_frames(&mut decoder)?;

    Ok(())
}

fn sample_layout_of_frame(frame: &frame::Video) -> SampleLayout {
    SampleLayout {
        channels: 3,
        channel_stride: 1,
        width: frame.width(),
        width_stride: 3,
        height: frame.height(),
        height_stride: frame.stride(0),
    }
}