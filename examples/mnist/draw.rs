use eframe::egui::{self, Color32, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Vec2};
use flate2::read::GzDecoder;
use meuron::{
    NeuralNetwork, NetworkType, Layers,
    DenseLayer, ReLU, Softmax,
    CrossEntropy, SGD,
    TrainOptions,
};
use ndarray::Array2;

use std::fs::{self, File};
use std::io::{self, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

type MnistNet = NeuralNetwork<
    NetworkType![DenseLayer<ReLU>, DenseLayer<ReLU>, DenseLayer<Softmax>],
    CrossEntropy,
>;

const GRID: usize = 28;
const SCALE: f32 = 16.0;
const BRUSH: f32 = 1.6;
const MODEL_PATH: &str = "./examples/mnist/mnist_model_draw.bin";
const DATA_DIR: &str = "./examples/mnist/data";
const MIRROR: &str = "https://systemds.apache.org/assets/datasets/mnist";
const FILES: &[&str] = &[
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
];

enum TrainMsg {
    Epoch { epoch: usize, total: usize, loss: f32, val_loss: Option<f32> },
    Accuracy(f32),
    Done,
    Error(String),
}

enum AppState {
    Idle,
    Training {
        rx: mpsc::Receiver<TrainMsg>,
        epochs_done: usize,
        total: usize,
        losses: Vec<f32>,
        val_losses: Vec<f32>,
        has_val: bool,
        status: String,
    },
    Drawing {
        nn: MnistNet,
        canvas: [[f32; GRID]; GRID],
        predictions: [f32; 10],
        best: Option<usize>,
    },
}

struct App {
    state: AppState,
    epoch_count: usize,
    batch_size: usize,
    learning_rate: f32,
    validation_split: f32,
    stop_at_loss_enabled: bool,
    stop_at_loss: f32,
    last_accuracy: Option<f32>,
}

impl App {
    fn new() -> Self {
        let state = if PathBuf::from(MODEL_PATH).exists() {
            match NeuralNetwork::load(MODEL_PATH, CrossEntropy) {
                Ok(nn) => AppState::Drawing {
                    nn,
                    canvas: [[0.0; GRID]; GRID],
                    predictions: [0.0; 10],
                    best: None,
                },
                Err(_) => AppState::Idle,
            }
        } else {
            AppState::Idle
        };

        Self {
            state,
            epoch_count: 25,
            batch_size: 4096,
            learning_rate: 0.05,
            validation_split: 0.1,
            stop_at_loss_enabled: false,
            stop_at_loss: 0.05,
            last_accuracy: None,
        }
    }

    fn start_training(
        &mut self,
        epochs: usize,
        batch_size: usize,
        lr: f32,
        val_split: f32,
        stop_loss: Option<f32>,
    ) {
        let (tx, rx) = mpsc::channel();
        let has_val = val_split > 0.0;

        std::thread::spawn(move || {
            let data_dir = PathBuf::from(DATA_DIR);

            if let Err(e) = ensure_mnist(&data_dir) {
                let _ = tx.send(TrainMsg::Error(e.to_string()));
                return;
            }

            let (images, labels) = match load_mnist(&data_dir, "train") {
                Ok(d) => d,
                Err(e) => {
                    let _ = tx.send(TrainMsg::Error(e.to_string()));
                    return;
                }
            };

            let mut nn: MnistNet = NeuralNetwork::new(
                Layers![
                    DenseLayer::new(28 * 28, 500, ReLU),
                    DenseLayer::new(500, 128, ReLU),
                    DenseLayer::new(128, 10, Softmax)
                ],
                CrossEntropy,
            );

            let tx2 = tx.clone();
            nn.train(
                images,
                labels,
                SGD::new(lr),
                TrainOptions::new()
                    .epochs(epochs)
                    .batch_size(batch_size)
                    .validation_split(val_split)
                    .callback(move |epoch, total, loss, val_loss| {
                        let keep_going = stop_loss.map_or(true, |t| loss > t);
                        tx2.send(TrainMsg::Epoch { epoch, total, loss, val_loss }).is_ok()
                            && keep_going
                    }),
            );

            let (test_images, test_labels) =
                match load_mnist(&PathBuf::from(DATA_DIR), "t10k") {
                    Ok(d) => d,
                    Err(_) => {
                        let _ = tx.send(TrainMsg::Done);
                        return;
                    }
                };

            let n = test_images.shape()[0];
            let pred_arr = nn.predict(test_images);
            let argmax = |row: ndarray::ArrayView1<f32>| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };
            let correct = (0..n)
                .filter(|&i| argmax(pred_arr.row(i)) == argmax(test_labels.row(i)))
                .count();

            let _ = tx.send(TrainMsg::Accuracy(correct as f32 / n as f32));
            let _ = nn.save(MODEL_PATH);
            let _ = tx.send(TrainMsg::Done);
        });

        self.state = AppState::Training {
            rx,
            epochs_done: 0,
            total: epochs,
            losses: Vec::new(),
            val_losses: Vec::new(),
            has_val,
            status: "Downloading MNIST...".into(),
        };
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

        if let AppState::Training {
            rx,
            epochs_done,
            total,
            losses,
            val_losses,
            status,
            ..
        } = &mut self.state
        {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    TrainMsg::Epoch { epoch, total: t, loss, val_loss } => {
                        *epochs_done = epoch;
                        *total = t;
                        losses.push(loss);
                        if let Some(vl) = val_loss {
                            val_losses.push(vl);
                        }
                        *status = match val_loss {
                            Some(vl) => format!(
                                "Epoch {epoch}/{t}  train={loss:.4}  val={vl:.4}"
                            ),
                            None => format!("Epoch {epoch}/{t}  loss={loss:.6}"),
                        };
                    }
                    TrainMsg::Accuracy(acc) => {
                        self.last_accuracy = Some(acc);
                    }
                    TrainMsg::Done => {
                        if let Ok(nn) = NeuralNetwork::load(MODEL_PATH, CrossEntropy) {
                            self.state = AppState::Drawing {
                                nn,
                                canvas: [[0.0; GRID]; GRID],
                                predictions: [0.0; 10],
                                best: None,
                            };
                            return;
                        }
                    }
                    TrainMsg::Error(e) => {
                        *status = format!("Error: {e}");
                    }
                }
            }
            ctx.request_repaint();
        }

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(Color32::from_gray(18))
                    .inner_margin(egui::Margin::same(24)),
            )
            .show(ctx, |ui| match &mut self.state {
                AppState::Idle => draw_idle(ui, self),
                AppState::Training { .. } => draw_training(ui, self),
                AppState::Drawing { .. } => draw_drawing(ui, self),
            });
    }
}

fn draw_idle(ui: &mut egui::Ui, app: &mut App) {
    ui.vertical_centered(|ui| {
        ui.add_space(40.0);
        ui.label(RichText::new("MNIST Digit Classifier").size(24.0).strong());
        ui.add_space(24.0);

        egui::Grid::new("settings")
            .num_columns(2)
            .spacing([16.0, 10.0])
            .show(ui, |ui| {
                ui.label("Epochs");
                ui.add(egui::DragValue::new(&mut app.epoch_count).range(1..=500));
                ui.end_row();

                ui.label("Batch size");
                ui.add(egui::DragValue::new(&mut app.batch_size).range(64..=16384));
                ui.end_row();

                ui.label("Learning rate");
                ui.add(
                    egui::DragValue::new(&mut app.learning_rate)
                        .speed(0.001)
                        .range(0.0001..=1.0)
                        .fixed_decimals(4),
                );
                ui.end_row();

                ui.label("Validation split");
                ui.add(
                    egui::Slider::new(&mut app.validation_split, 0.0..=0.5)
                        .fixed_decimals(2)
                        .text(""),
                );
                ui.end_row();

                ui.label("Train until loss ≤");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut app.stop_at_loss_enabled, "");
                    ui.add_enabled(
                        app.stop_at_loss_enabled,
                        egui::DragValue::new(&mut app.stop_at_loss)
                            .speed(0.001)
                            .range(0.001..=10.0)
                            .fixed_decimals(4),
                    );
                    if !app.stop_at_loss_enabled {
                        ui.label(
                            RichText::new("disabled").size(12.0).color(Color32::from_gray(90)),
                        );
                    }
                });
                ui.end_row();
            });

        ui.add_space(8.0);

        if app.validation_split > 0.0 {
            ui.label(
                RichText::new(format!(
                    "{:.0}% held out for validation, {:.0}% used for training",
                    app.validation_split * 100.0,
                    (1.0 - app.validation_split) * 100.0,
                ))
                .size(11.0)
                .color(Color32::from_gray(100)),
            );
        } else {
            ui.label(
                RichText::new("No validation split — full dataset used for training")
                    .size(11.0)
                    .color(Color32::from_gray(100)),
            );
        }

        ui.add_space(24.0);

        if ui
            .add_sized(
                [180.0, 40.0],
                egui::Button::new(RichText::new("Train").size(16.0))
                    .fill(Color32::from_rgb(40, 140, 80)),
            )
            .clicked()
        {
            let e = app.epoch_count;
            let bs = app.batch_size;
            let lr = app.learning_rate;
            let vs = app.validation_split;
            let sl = app.stop_at_loss_enabled.then_some(app.stop_at_loss);
            app.start_training(e, bs, lr, vs, sl);
        }
    });
}

fn draw_training(ui: &mut egui::Ui, app: &mut App) {
    let (epochs_done, total, losses, val_losses, has_val, status) = match &app.state {
        AppState::Training {
            epochs_done,
            total,
            losses,
            val_losses,
            has_val,
            status,
            ..
        } => (
            *epochs_done,
            *total,
            losses.clone(),
            val_losses.clone(),
            *has_val,
            status.clone(),
        ),
        _ => return,
    };

    ui.vertical_centered(|ui| {
        ui.add_space(20.0);
        ui.label(RichText::new("Training…").size(20.0).strong());
        ui.add_space(10.0);
        ui.label(RichText::new(&status).size(13.0).color(Color32::from_gray(180)));
        ui.add_space(12.0);

        let progress = if total > 0 {
            epochs_done as f32 / total as f32
        } else {
            0.0
        };
        ui.add(
            egui::ProgressBar::new(progress)
                .desired_width(480.0)
                .text(format!("{epochs_done} / {total}")),
        );

        if losses.len() > 1 {
            ui.add_space(20.0);

            let all_vals = losses.iter().chain(val_losses.iter()).copied();
            let y_max = all_vals
                .clone()
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.01);
            let y_min = all_vals.fold(f32::INFINITY, f32::min).max(0.0) * 0.95;

            let train_pts: Vec<[f64; 2]> = losses
                .iter()
                .enumerate()
                .map(|(i, &l)| [i as f64 + 1.0, l as f64])
                .collect();

            let val_pts: Vec<[f64; 2]> = val_losses
                .iter()
                .enumerate()
                .map(|(i, &l)| [i as f64 + 1.0, l as f64])
                .collect();

            egui_plot::Plot::new("loss_plot")
                .height(260.0)
                .width(480.0)
                .include_y(y_min as f64)
                .include_y(y_max as f64 * 1.05)
                .x_axis_label("Epoch")
                .y_axis_label("Loss")
                .legend(egui_plot::Legend::default().position(egui_plot::Corner::RightTop))
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        egui_plot::Line::new(
                            "Train loss",
                            egui_plot::PlotPoints::from(train_pts),
                        )
                        .color(Color32::from_rgb(66, 135, 245))
                        .width(2.0),
                    );

                    if has_val && !val_pts.is_empty() {
                        plot_ui.line(
                            egui_plot::Line::new(
                                "Val loss",
                                egui_plot::PlotPoints::from(val_pts),
                            )
                            .color(Color32::from_rgb(255, 160, 50))
                            .width(2.0),
                        );
                    }
                });

            ui.add_space(8.0);
            if let (Some(best_train), last_train) = (
                losses.iter().copied().reduce(f32::min),
                losses.last().copied(),
            ) {
                let stats = if has_val && !val_losses.is_empty() {
                    let best_val = val_losses.iter().copied().reduce(f32::min).unwrap_or(0.0);
                    format!(
                        "best train {best_train:.4}   best val {best_val:.4}   current {:.4}",
                        last_train.unwrap_or(0.0)
                    )
                } else {
                    format!(
                        "best {best_train:.4}   current {:.4}",
                        last_train.unwrap_or(0.0)
                    )
                };
                ui.label(RichText::new(stats).size(11.0).color(Color32::from_gray(110)));
            }
        }
    });
}

fn draw_drawing(ui: &mut egui::Ui, app: &mut App) {
    let (nn, canvas, predictions, best) = match &mut app.state {
        AppState::Drawing { nn, canvas, predictions, best } => (nn, canvas, predictions, best),
        _ => return,
    };

    let last_accuracy = app.last_accuracy;

    if let Some(acc) = last_accuracy {
        ui.label(
            RichText::new(format!("Test accuracy: {:.2}%", acc * 100.0))
                .size(13.0)
                .color(Color32::from_gray(140)),
        );
        ui.add_space(8.0);
    }

    let mut retrain = false;

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(
                RichText::new("Draw (Left Click paint - Right Click erase)")
                    .size(14.0)
                    .color(Color32::from_gray(160)),
            );
            ui.add_space(8.0);

            let canvas_px = GRID as f32 * SCALE;
            let (rect, response) =
                ui.allocate_exact_size(Vec2::splat(canvas_px), Sense::click_and_drag());

            let mut changed = false;

            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(pos) = response.interact_pointer_pos() {
                    let local = pos - rect.min;
                    let cp = Pos2::new(local.x / SCALE, local.y / SCALE);
                    for y in 0..GRID {
                        for x in 0..GRID {
                            let dx = x as f32 + 0.5 - cp.x;
                            let dy = y as f32 + 0.5 - cp.y;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < BRUSH {
                                let ink = 1.0 - (dist / BRUSH).powi(2);
                                canvas[y][x] = (canvas[y][x] + ink * 0.6).min(1.0);
                            }
                        }
                    }
                    changed = true;
                }
            }

            if response.dragged_by(egui::PointerButton::Secondary) {
                if let Some(pos) = response.interact_pointer_pos() {
                    let local = pos - rect.min;
                    let cp = Pos2::new(local.x / SCALE, local.y / SCALE);
                    let radius = BRUSH * 1.5;
                    for y in 0..GRID {
                        for x in 0..GRID {
                            let dx = x as f32 + 0.5 - cp.x;
                            let dy = y as f32 + 0.5 - cp.y;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < radius {
                                let fade = (dist / radius).powi(2);
                                canvas[y][x] = (canvas[y][x] * fade).max(0.0);
                            }
                        }
                    }
                    changed = true;
                }
            }

            if changed {
                let flat: Vec<f32> = canvas
                    .iter()
                    .flat_map(|row: &[f32; GRID]| row.iter().copied())
                    .collect();
                let input = Array2::from_shape_vec((1, GRID * GRID), flat).unwrap();
                let arr: Array2<f32> = nn.predict(input);
                let mut bi = 0;
                let mut bv = 0.0_f32;
                for i in 0..10 {
                    predictions[i] = arr[[0, i]];
                    if arr[[0, i]] > bv {
                        bv = arr[[0, i]];
                        bi = i;
                    }
                }
                *best = if bv > 0.02 { Some(bi) } else { None };
            }

            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 6.0, Color32::from_gray(8));

            for y in 0..GRID {
                for x in 0..GRID {
                    let v = canvas[y][x];
                    if v > 0.005 {
                        let b = (v.powf(0.7) * 255.0) as u8;
                        let min =
                            rect.min + Vec2::new(x as f32 * SCALE, y as f32 * SCALE);
                        painter.rect_filled(
                            Rect::from_min_size(min, Vec2::splat(SCALE)),
                            0.0,
                            Color32::from_rgb(b, b, b),
                        );
                    }
                }
            }

            let gc = Color32::from_rgba_premultiplied(255, 255, 255, 6);
            for i in 1..GRID {
                let off = i as f32 * SCALE;
                painter.line_segment(
                    [rect.min + Vec2::new(off, 0.0), rect.min + Vec2::new(off, canvas_px)],
                    Stroke::new(0.5, gc),
                );
                painter.line_segment(
                    [rect.min + Vec2::new(0.0, off), rect.min + Vec2::new(canvas_px, off)],
                    Stroke::new(0.5, gc),
                );
            }

            painter.rect_stroke(
                rect,
                6.0,
                Stroke::new(1.5, Color32::from_gray(55)),
                StrokeKind::Middle,
            );

            ui.add_space(12.0);

            ui.horizontal(|ui| {
                if ui
                    .add_sized(
                        [canvas_px / 2.0 - 4.0, 32.0],
                        egui::Button::new(RichText::new("Clear").size(14.0))
                            .fill(Color32::from_gray(35)),
                    )
                    .clicked()
                {
                    *canvas = [[0.0; GRID]; GRID];
                    *predictions = [0.0; 10];
                    *best = None;
                }

                if ui
                    .add_sized(
                        [canvas_px / 2.0 - 4.0, 32.0],
                        egui::Button::new(RichText::new("Retrain").size(14.0))
                            .fill(Color32::from_gray(35)),
                    )
                    .clicked()
                {
                    retrain = true;
                }
            });
        });

        ui.add_space(28.0);

        ui.vertical(|ui| {
            ui.label(
                RichText::new("Confidence")
                    .size(14.0)
                    .color(Color32::from_gray(160)),
            );
            ui.add_space(8.0);

            let bar_w = 180.0_f32;

            for digit in 0..10usize {
                let conf = predictions[digit];
                let is_best = *best == Some(digit);

                ui.horizontal(|ui| {
                    ui.add_sized(
                        [22.0, 22.0],
                        egui::Label::new(
                            RichText::new(format!("{digit}"))
                                .size(15.0)
                                .monospace()
                                .color(if is_best {
                                    Color32::from_rgb(255, 215, 50)
                                } else {
                                    Color32::from_gray(130)
                                }),
                        ),
                    );

                    let (bar_rect, _) =
                        ui.allocate_exact_size(Vec2::new(bar_w, 20.0), Sense::hover());
                    let p = ui.painter_at(bar_rect);
                    p.rect_filled(bar_rect, 4.0, Color32::from_gray(28));

                    let fill_w = (conf * bar_w).clamp(0.0, bar_w);
                    if fill_w > 0.5 {
                        p.rect_filled(
                            Rect::from_min_size(bar_rect.min, Vec2::new(fill_w, 20.0)),
                            4.0,
                            if is_best {
                                Color32::from_rgb(40, 200, 90)
                            } else {
                                Color32::from_rgb(55, 110, 195)
                            },
                        );
                    }

                    ui.add_sized(
                        [52.0, 20.0],
                        egui::Label::new(
                            RichText::new(format!("{:5.1}%", conf * 100.0))
                                .size(12.0)
                                .monospace()
                                .color(Color32::from_gray(160)),
                        ),
                    );
                });

                ui.add_space(4.0);
            }

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(12.0);

            match best {
                Some(d) => {
                    ui.label(
                        RichText::new(format!("{d}"))
                            .size(80.0)
                            .strong()
                            .color(Color32::from_rgb(40, 200, 90)),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}% confident", predictions[*d] * 100.0))
                            .size(13.0)
                            .color(Color32::from_gray(140)),
                    );
                }
                None => {
                    ui.label(
                        RichText::new("-")
                            .size(80.0)
                            .color(Color32::from_gray(50)),
                    );
                    ui.label(
                        RichText::new("draw something")
                            .size(13.0)
                            .color(Color32::from_gray(80)),
                    );
                }
            }
        });
    });

    if retrain {
        app.state = AppState::Idle;
    }
}

fn ensure_mnist(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    for &gz_name in FILES {
        let dest = dir.join(gz_name.strip_suffix(".gz").unwrap());
        if dest.exists() {
            continue;
        }
        let url = format!("{}/{}", MIRROR, gz_name);
        let response = ureq::get(&url)
            .call()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let mut body = response.into_body();
        let mut gz = GzDecoder::new(body.as_reader());
        let mut out = BufWriter::new(File::create(&dest)?);
        io::copy(&mut gz, &mut out)?;
    }
    Ok(())
}

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_mnist(dir: &Path, prefix: &str) -> io::Result<(Array2<f32>, Array2<f32>)> {
    ensure_mnist(dir)?;
    let mut img_f = File::open(dir.join(format!("{}-images-idx3-ubyte", prefix)))?;
    let mut lbl_f = File::open(dir.join(format!("{}-labels-idx1-ubyte", prefix)))?;

    let _ = read_u32(&mut img_f)?;
    let n = read_u32(&mut img_f)? as usize;
    let r = read_u32(&mut img_f)? as usize;
    let c = read_u32(&mut img_f)? as usize;
    let _ = read_u32(&mut lbl_f)?;
    let nl = read_u32(&mut lbl_f)? as usize;
    assert_eq!(n, nl);

    let mut raw_img = vec![0u8; n * r * c];
    let mut raw_lbl = vec![0u8; n];
    img_f.read_exact(&mut raw_img)?;
    lbl_f.read_exact(&mut raw_lbl)?;

    let images = Array2::from_shape_vec(
        (n, r * c),
        raw_img.into_iter().map(|x| x as f32 / 255.0).collect(),
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let labels = Array2::from_shape_vec(
        (n, 10),
        raw_lbl
            .into_iter()
            .flat_map(|l| {
                let mut oh = [0.0f32; 10];
                oh[l as usize] = 1.0;
                oh
            })
            .collect(),
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok((images, labels))
}


fn main() {
    eframe::run_native(
        "Meuron - MNIST Drawing Example",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([720.0, 580.0])
                .with_resizable(true)
                .with_title("Meuron - MNIST Drawing Example"),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
    .unwrap();
}
