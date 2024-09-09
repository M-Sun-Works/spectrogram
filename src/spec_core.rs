/*
 * Copyright (C) Simon Werner, 2019.
 *
 * A Rust port of the original C++ code by Christian Briones, 2013.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

 use std::sync::Arc;
 use std::{cmp::min, f32};
 use num_traits::real::Real;
 
 use crate::{Spectrogram, WindowFn};
 use rustfft::{num_complex::Complex, FftPlanner};
 
 #[cfg(feature = "rayon")]
 use rayon::prelude::*;
 
 use parquet::file::reader::{FileReader, SerializedFileReader};
 use parquet::record::RowAccessor;
 use std::fs::File;

 use std::f64::consts::PI;

use std::io::{self, BufReader, BufWriter};
use flate2::read::GzDecoder;
use std::io::Read;
use image::{GrayImage, Luma};
use std::path::Path;


 ///
 /// This contains all the initialised data.  This can then produce the spectrogram,
 /// and if necessary, save it to the filesystem as a PNG image.
 ///
 /// This `Spectrograph` is created by `SpecOptionsBuilder`.
 ///
 /// # Example
 ///
 /// ```Rust
 ///   let mut spectrograph = SpecOptionsBuilder::new(2048)
 ///     .load_data_from_file(&std::path::Path::new(wav_file))?
 ///     .build();
 ///
 ///   // Compute the spectrogram.  Need export it using `to_png()` or simlar.
 ///   spectrograph.compute();
 /// ```
 ///
 pub struct SpecCompute {
     num_bins: usize,     // The num of fft bins in the spectrogram.
     data: Vec<f32>,      // The time domain data for the FFT.  Normalised to meet -1.0..1.0.
     window_fn: WindowFn, // The Window Function to apply to each fft window.
     step_size: usize, // The step size in the window function, must be less than the window function
     fft_fn: Arc<dyn rustfft::Fft<f32>>, // An FFT function that can be used to compute the FFT.
 
     #[cfg(feature = "rayon")]
     gen_fft_fn: Arc<dyn Sync + Send + Fn() -> Arc<dyn rustfft::Fft<f32>>>, // An FFT generator for each thread
 }
 
 pub trait LPFandDecimate {
    fn cic_filter<'a>(&mut self, input: &'a mut [f64]) -> &'a mut [f64];
    fn butter_filter<'a>(&mut self, input: &'a mut [f64]) -> &'a mut [f64];
 }
 pub struct CicfiltDecimate<const STAGES: usize, const RATIO: usize> {
     integrator: [f64; STAGES],
     comb: [f64; STAGES],
     delay_idx: usize,
     normalization_bits: usize,
 }
 
 impl<const STAGES: usize, const RATIO: usize> CicfiltDecimate<STAGES, RATIO> {
     pub fn new() -> Self {
         const {
             assert!(STAGES > 0);
             assert!(RATIO > 0);
             assert!(RATIO.is_power_of_two());
         }
         let bits = ((RATIO as f64).log2() + 0.5).floor() as usize;
         Self {
             integrator: [0.0; STAGES],
             comb: [0.0; STAGES],
             delay_idx: 0,
             normalization_bits: bits * STAGES,
         }
     }
 }
 impl<const STAGES: usize, const RATIO: usize> LPFandDecimate for CicfiltDecimate<STAGES, RATIO> {
    fn cic_filter<'a>(&mut self, input: &'a mut [f64]) -> &'a mut [f64] {

        let mut in_idx = 0;
        let mut out_idx = 0;
        while in_idx < input.len() {
            let mut sample = input[in_idx];
            in_idx += 1;

            self.integrator.iter_mut().for_each(|x: &mut f64| {
                *x += sample as f64;
                sample = *x as f64;
            });

            self.delay_idx += 1;
            if self.delay_idx != RATIO {
                continue;
            }
            self.delay_idx = 0;

            self.comb.iter_mut().for_each(|x: &mut f64| {
                let zi = *x;
                *x = sample as f64;
                sample -= zi as f64;
            });

            sample /= 2f64.powi(self.normalization_bits as i32);
            input[out_idx] = sample;
            out_idx += 1;
        }
        &mut input[..out_idx]
    }

    fn butter_filter<'a>(&mut self, input: &'a mut [f64]) -> &'a mut [f64] {
        let mut in_idx = 0;
        let mut out_idx = 0;
        while in_idx < input.len() {
            let mut sample = input[in_idx];
            in_idx += 1;

            self.integrator.iter_mut().for_each(|x: &mut f64| {
                sample += *x;
                *x = sample;
            });

            self.delay_idx += 1;
            if self.delay_idx != RATIO {
                continue;
            }
            self.delay_idx = 0;

            self.comb.iter_mut().for_each(|x: &mut f64| {
                let zi = *x;
                *x = sample;
                sample -= zi;
            });

            sample /= 2f64.powi(self.normalization_bits as i32);
            input[out_idx] = sample;
            out_idx += 1;
        }
        &mut input[..out_idx]
    }
}

 
 fn quadrature_mix(signal: &[f64], f_c: f64, sample_rate: f64, length: usize) -> Vec<Complex<f64>> {
     let mixer: Vec<Complex<f64>> = (0..length).map(|n| {
         let t = n as f64 / sample_rate;
         let angle = -2.0 * PI * f_c * t;
         Complex::new(angle.cos(), angle.sin()) // This is the complex exponential
     }).collect();
     signal.iter().zip(mixer.iter())
         .map(|(&x, &m)| Complex::new(x, 0.0) * m) 
         .collect()
 }

 fn decompress_csv(input_file: &str) -> io::Result<String> {
    // Open the .gz file for reading
    let file = File::open(input_file)?;
    let mut decoder = GzDecoder::new(file);

    // Read the decompressed data into a String
    let mut decompressed_data = String::new();
    decoder.read_to_string(&mut decompressed_data)?;

    Ok(decompressed_data)
}

use csv::ReaderBuilder;

fn parse_csv_data(data: &str, column_index: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut rdr = ReaderBuilder::new().from_reader(data.as_bytes());
    let mut result = Vec::new();

    for record in rdr.records() {
        let record = record?;
        if let Some(field) = record.get(column_index) {
            if let Ok(value) = field.parse::<f32>() {
                result.push(value);
            }
        }
    }

    Ok(result)
}




 impl SpecCompute {
     /// Create a new Spectrograph from data.
     ///
     /// **You probably want to use [SpecOptionsBuilder] instead.**
     pub fn new(num_bins: usize, step_size: usize, data: Vec<f32>, window_fn: WindowFn) -> Self {
         // Compute the FFT plan generator and a single FFT plan, if supporting parallel FFTs.
         #[cfg(feature = "rayon")]
         {
             let gen_fft_fn = Arc::new(move || {
                 let mut planner = FftPlanner::new();
                 planner.plan_fft_forward(num_bins)
             });
             let fft_fn = gen_fft_fn();
             SpecCompute {
                 num_bins,
                 step_size,
                 data,
                 window_fn,
                 fft_fn,
                 gen_fft_fn,
             }
         }
 
         // Compute a single FFT plan, if not supporting parallel FFTs.
         #[cfg(not(feature = "rayon"))]
         {
             let mut planner = FftPlanner::<f32>::new();
             let fft_fn = planner.plan_fft_forward(num_bins);
             SpecCompute {
                 num_bins,
                 step_size,
                 data,
                 window_fn,
                 fft_fn,
             }
         }
     }
 
     ///
     /// Update the sample data with a new set.  Note, none of the settings
     /// from the builder are applied, all the samples are used in their raw form.
     ///
     pub fn set_data(&mut self, data: Vec<f32>) {
         self.data = data;
     }
 
     ///
     /// Do the discrete fourier transform to create the spectrogram.
     ///
     /// # Arguments
     ///
     ///  * `n_fft` - How many fourier transform frequency bins to use. Must be a
     ///                 power of 2.
     ///
     pub fn compute(&mut self) -> Spectrogram {
         let width = (self.data.len() - self.num_bins) / self.step_size;
         let height = self.num_bins / 2;
 
         let mut spec = vec![0.0; self.num_bins * width];
 
         let mut p = 0; // Index to the beginning of the window
 
         // Once, Allocate buffers that will be used for computation
         let mut inplace_buf: Vec<Complex<f32>> = vec![Complex::new(0., 0.); self.num_bins];
         let mut scratch_buf: Vec<Complex<f32>> =
             vec![Complex::new(0., 0.); self.fft_fn.get_inplace_scratch_len()];
 
         // Create slices into the buffers backing the Vecs to be reused on each loop
         let inplace_slice = &mut inplace_buf[..];
         let scratch_slice = &mut scratch_buf[..];
 
         for w in 0..width {
             // Extract the next `num_bins` complex floats into the FFT inplace compute buffer
             self.data[p..]
                 .iter()
                 .take(self.num_bins)
                 .enumerate()
                 .map(|(i, val)| val * (self.window_fn)(i, self.num_bins)) // Apply the window function
                 .map(|val| Complex::new(val, 0.0))
                 .zip(inplace_slice.iter_mut())
                 .for_each(|(c, v)| *v = c);
 
             // Call out to rustfft to actually compute the FFT
             // This will take the inplace_slice as input, use scratch_slice during computation, and write FFT back into inplace_slice
             let inplace = &mut inplace_slice[..min(self.num_bins, self.data.len() - p)];
             self.fft_fn.process_with_scratch(inplace, scratch_slice);
 
             // Normalize the spectrogram and write to the output
             inplace
                 .iter()
                 .take(height)
                 .rev()
                 .map(|c_val| c_val.norm())
                 .zip(spec[w..].iter_mut().step_by(width))
                 .for_each(|(a, b)| *b = a);
 
             p += self.step_size;
         }
 
         Spectrogram {
             spec,
             width,
             height,
         }
     }
 
     ///
     /// Do the discrete fourier transform to create the spectrogram.
     ///
     /// This function will use rayon to parallelize the FFT computation.
     /// It may create more FFT plans than there are threads, but will reuse them
     /// if called multiple times.
     ///
     ///
     /// # Arguments
     ///
     ///  * `self` -  `self` is immutable so that the FFT plans can be shared between threads.
     ///               Since the FFT plans are not thread safe, they are wrapped in a Mutex.
     ///
     ///  * `data` -  The time domain data to compute the spectrogram from.
     ///              `data` is not preprocessed other than windowing and
     ///               casting to complex.
     ///
     #[cfg(feature = "rayon")]
     pub fn par_compute(&self, data: Option<&[f32]>) -> Spectrogram {
         let data = data.unwrap_or(self.data.as_slice());
         let width = (data.len() - self.num_bins) / self.step_size;
         let height = self.num_bins / 2;
 
         // Compute the spectrogram in parallel steps via rayon
         let spec_cols: Vec<_> = (0..width)
             .into_par_iter()
             .map_init(|| {
                 let fft_fn = (self.gen_fft_fn)();
                 let scratch_buf = vec![Complex::new(0., 0.); fft_fn.get_inplace_scratch_len()];
                     (
                         fft_fn,
                         vec![Complex::new(0., 0.); self.num_bins],
                         scratch_buf,
                     )
                 }, | (fft_fn, inplace_buf, scratch_buf), w| {
                     // Index to the beginning of the window
                     let window_index = w * self.step_size;
 
                     // Extract the next `num_bins` complex floats into the FFT inplace compute buffer
                     data[(w * self.step_size)..]
                     .iter()
                     .take(self.num_bins)
                     .enumerate()
                     .map(|(i, val)| Complex::new(val * (self.window_fn)(i, self.num_bins), 0.))
                     .zip(inplace_buf.iter_mut())
                     .for_each(|(c, v)| *v = c);
 
                     // Create slices into the buffers backing the Vecs to be reused on each loop
                     let inplace_slice = inplace_buf.as_mut_slice();
                     let scratch_slice = scratch_buf.as_mut_slice();
 
                     // Call out to rustfft to actually compute the FFT
                     // This will take the inplace_slice as input, use scratch_slice during computation, and write FFT back into inplace_slice
                     let inplace =
                         &mut inplace_slice[..min(self.num_bins, data.len() - window_index)];
                     fft_fn.process_with_scratch(inplace, scratch_slice);
 
                     // Normalize the spectrogram and write to the output
                     let spec_col =   inplace
                         .iter()
                         .take(height)
                         .rev()
                         .map(|c_val| c_val.norm())
                         .collect::<Vec<_>>();
 
                     spec_col
                 })
             .collect();
 
 
         // Transpose the columns into row major order
         let spec = (0..height)
             .into_par_iter()
             .flat_map_iter(|i| spec_cols.iter().map(move |row| row[i]))
             .collect::<Vec<_>>();
 
         Spectrogram {
             spec,
             width,
             height,
         }
     }
 }
 
#[cfg(test)]
mod tests {
    use crate::ColourGradient;
    use crate::FrequencyScale;
    use super::*;

    #[test]
    fn asdf() {
        let _data = [0];
    }
    #[test]
    fn main() {
        // Read and decompress the data
        let decompressed_data = decompress_csv("/Users/maxwellsun/Desktop/Workspace/D/deepdata.csv.gz").unwrap();
    
        // Parameters
        let column_index = 1; // Change this index to the column you want to read
        let data = parse_csv_data(&decompressed_data, column_index).unwrap();
    
        // Other parameters
        let num_bins = 1024;
        let step_size = 128;
        let window_fn = |i: usize, n: usize| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos())
        };
    
        // Create the spectrogram computation object
        let mut spec_compute = SpecCompute::new(num_bins, step_size, data.clone(), window_fn);
    
        // Compute the spectrogram
        let spectrogram: &mut Spectrogram = &mut spec_compute.compute();
        let width = (data.len() - num_bins) / step_size;
        let height = num_bins / 2;
        let gradient: &mut ColourGradient = &mut ColourGradient::twilight_theme();
        spectrogram.to_png(Path::new("/Users/maxwellsun/Desktop/Workspace/spectrogram/spectrogram.png"), FrequencyScale::Linear, gradient, width, height,None,None);
        // use image::{ImageBuffer, Rgba};
    
        // // Create the colour gradient
        // let gradient = ColourGradient::create(ColourTheme::Twilight);
    
        // // Create an image buffer
        // let mut img = ImageBuffer::new(width as u32, height as u32);
    
        // // Find the min and max values in the spectrogram for normalization
        // let min_value = spectrogram.spec.iter().cloned().fold(f32::INFINITY, f32::min);
        // let max_value = spectrogram.spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
        // // Normalize and convert to gradient colors
        // for (i, &value) in spectrogram.spec.iter().enumerate() {
        //     let x = (i % width) as u32;
        //     let y = (i / width) as u32;
    
        //     if x < img.width() && y < img.height() {
        //         // Normalize the value to the range [0.0, 1.0]
        //         let normalized_value = (value - min_value) / (max_value - min_value);
    
        //         // Get the color from the gradient
        //         let color = gradient.get_colour(normalized_value);
    
        //         // Store the color in the image
        //         img.put_pixel(x, y, Rgba([color.r, color.g, color.b, color.a]));
        //     }
        // }
    
        // // Save the image to a file
        // let path = Path::new("spectrogram_twilight.png");
        // img.save(path).expect("Failed to save image");
    }
}