export type Method = "rgb" | "chrominance" | "perceptual" | "combined" | "adversarial";
export type Metrics = {
  psnr: number;
  ssim: number;
  delta_e_2000: number;
  mean_absolute_rgb_difference: number;
  maximum_pixel_difference: number;
  changed_pixel_percentage: number;
};
export type ProcessResult = {
  image_id: string;
  filename: string;
  original_url: string;
  modified_url: string;
  difference_url: string;
  width: number;
  height: number;
  original_size: number;
  modified_size: number;
  processing_time_ms: number;
  metrics: Metrics;
};
export type HistoryItem = {
  id: string;
  filename: string;
  date: string;
  method: Method;
  budget: number;
  duration_ms: number;
  metrics: Metrics;
  width: number;
  height: number;
};
