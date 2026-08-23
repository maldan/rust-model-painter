use mega_render::{AoMethod, PostProcessSettings, Scene, SsgiQuality};
use mega_ui::{ScrollAxes, TextStyle, Ui};

/// Post-process panel. Returns true if the UI wants continuous repaint.
pub fn post_effects_panel(ui: &mut Ui, post: &mut PostProcessSettings, scene: &mut Scene) -> bool {
    let mut keep = false;
    let size = ui.available_size();
    ui.scroll_area("Effects", size, ScrollAxes::Vertical, |ui| {
        ui.label_styled(
            "Effects",
            TextStyle {
                color: [0.85, 0.75, 0.35, 1.0],
                size: 16.0,
            },
        );
        ui.separator();

        ui.collapsing_header("Env Map", |ui| {
            if ui.checkbox("Enabled", &mut post.env.enabled).changed() {
                keep = true;
            }
            ui.label("Equirect reflections + skybox");
            ui.add_enabled(post.env.enabled, |ui| {
                ui.label("Intensity");
                if ui
                    .slider("env_intensity", &mut post.env.intensity, 0.0..=3.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Rotation Y (°)");
                if ui
                    .slider("env_rot", &mut post.env.rotation_y, 0.0..=360.0)
                    .changed()
                {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("AO", |ui| {
            if ui.checkbox("Enabled", &mut post.ao.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.ao.enabled, |ui| {
                let mut method = match post.ao.method {
                    AoMethod::Ssao => 0usize,
                    AoMethod::Gtao => 1,
                };
                ui.label("Method");
                if ui
                    .select("ao_method", &mut method, &["SSAO", "GTAO"])
                    .changed()
                {
                    post.ao.method = if method == 0 {
                        AoMethod::Ssao
                    } else {
                        AoMethod::Gtao
                    };
                    keep = true;
                }
                ui.label("Radius");
                if ui
                    .slider("ao_radius", &mut post.ao.radius, 0.1..=3.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Intensity");
                if ui
                    .slider("ao_intensity", &mut post.ao.intensity, 0.0..=2.0)
                    .changed()
                {
                    keep = true;
                }
                match post.ao.method {
                    AoMethod::Ssao => {
                        ui.label("Bias");
                        if ui
                            .slider("ao_bias", &mut post.ao.bias, 0.0..=0.2)
                            .changed()
                        {
                            keep = true;
                        }
                    }
                    AoMethod::Gtao => {
                        let mut dirs = post.ao.directions as f32;
                        let mut steps = post.ao.steps as f32;
                        ui.label("Directions");
                        if ui.slider("ao_dirs", &mut dirs, 2.0..=8.0).changed() {
                            post.ao.directions = dirs.round() as u32;
                            keep = true;
                        }
                        ui.label("Steps");
                        if ui.slider("ao_steps", &mut steps, 2.0..=12.0).changed() {
                            post.ao.steps = steps.round() as u32;
                            keep = true;
                        }
                        ui.label("Thickness");
                        if ui
                            .slider("ao_thickness", &mut post.ao.thickness, 0.2..=3.0)
                            .changed()
                        {
                            keep = true;
                        }
                    }
                }
            });
        });

        ui.collapsing_header("Contact Shadows", |ui| {
            if ui
                .checkbox("Enabled", &mut post.contact_shadow.enabled)
                .changed()
            {
                keep = true;
            }
            ui.add_enabled(post.contact_shadow.enabled, |ui| {
                ui.label("Length");
                if ui
                    .slider("cs_length", &mut post.contact_shadow.length, 0.05..=1.5)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Thickness");
                if ui
                    .slider(
                        "cs_thickness",
                        &mut post.contact_shadow.thickness,
                        0.01..=2.0,
                    )
                    .changed()
                {
                    keep = true;
                }
                ui.label("Intensity");
                if ui
                    .slider(
                        "cs_intensity",
                        &mut post.contact_shadow.intensity,
                        0.0..=2.0,
                    )
                    .changed()
                {
                    keep = true;
                }
                let mut samples = post.contact_shadow.samples as f32;
                ui.label("Samples");
                if ui.slider("cs_samples", &mut samples, 4.0..=32.0).changed() {
                    post.contact_shadow.samples = samples.round() as u32;
                    keep = true;
                }
                ui.label("Bias");
                if ui
                    .slider("cs_bias", &mut post.contact_shadow.bias, 0.0..=0.05)
                    .changed()
                {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("SSGI", |ui| {
            if ui.checkbox("Enabled", &mut post.ssgi.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.ssgi.enabled, |ui| {
                let mut q = match (post.ssgi.samples, post.ssgi.max_steps) {
                    (s, t) if s <= 4 && t <= 4 => 0usize,
                    (s, t) if s >= 12 || t >= 12 => 2,
                    _ => 1,
                };
                ui.label("Quality");
                if ui
                    .select(
                        "ssgi_quality",
                        &mut q,
                        &["Low (4×4)", "Medium (8×8)", "High (12×12)"],
                    )
                    .changed()
                {
                    SsgiQuality::ALL[q].apply(&mut post.ssgi);
                    keep = true;
                }
                ui.label("Radius");
                if ui
                    .slider("ssgi_radius", &mut post.ssgi.radius, 0.2..=4.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Thickness");
                if ui
                    .slider("ssgi_thickness", &mut post.ssgi.thickness, 0.02..=1.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Intensity");
                if ui
                    .slider("ssgi_intensity", &mut post.ssgi.intensity, 0.0..=5.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Energy");
                if ui
                    .slider("ssgi_energy", &mut post.ssgi.energy, 0.25..=3.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("2nd bounce");
                if ui
                    .slider("ssgi_2nd", &mut post.ssgi.second_bounce, 0.0..=1.5)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Ambient dim");
                if ui
                    .slider("ssgi_amb", &mut post.ssgi.ambient_dim, 0.0..=1.0)
                    .changed()
                {
                    keep = true;
                }
                if ui.checkbox("Temporal", &mut post.ssgi.temporal).changed() {
                    keep = true;
                }
                ui.add_enabled(post.ssgi.temporal, |ui| {
                    ui.label("History");
                    if ui
                        .slider("ssgi_hist", &mut post.ssgi.history, 0.5..=0.98)
                        .changed()
                    {
                        keep = true;
                    }
                });
            });
        });

        ui.collapsing_header("SSR", |ui| {
            if ui.checkbox("Enabled", &mut post.ssr.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.ssr.enabled, |ui| {
                ui.label("Max distance");
                if ui
                    .slider("ssr_dist", &mut post.ssr.max_distance, 0.5..=20.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Thickness");
                if ui
                    .slider("ssr_thickness", &mut post.ssr.thickness, 0.02..=1.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Intensity");
                if ui
                    .slider("ssr_intensity", &mut post.ssr.intensity, 0.0..=2.0)
                    .changed()
                {
                    keep = true;
                }
                let mut steps = post.ssr.max_steps as f32;
                ui.label("Steps");
                if ui.slider("ssr_steps", &mut steps, 8.0..=64.0).changed() {
                    post.ssr.max_steps = steps.round() as u32;
                    keep = true;
                }
                ui.label("Roughness cutoff");
                if ui
                    .slider("ssr_rough", &mut post.ssr.roughness_cutoff, 0.1..=1.0)
                    .changed()
                {
                    keep = true;
                }
                if ui.checkbox("Temporal", &mut post.ssr.temporal).changed() {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("Bloom", |ui| {
            if ui.checkbox("Enabled", &mut post.bloom.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.bloom.enabled, |ui| {
                ui.label("Threshold");
                if ui
                    .slider("bloom_th", &mut post.bloom.threshold, 0.0..=4.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Intensity");
                if ui
                    .slider("bloom_int", &mut post.bloom.intensity, 0.0..=2.0)
                    .changed()
                {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("DOF", |ui| {
            if ui.checkbox("Enabled", &mut post.dof.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.dof.enabled, |ui| {
                if ui
                    .checkbox("Auto focus", &mut post.dof.auto_focus)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Focus distance");
                if ui
                    .slider(
                        "dof_focus",
                        &mut scene.camera.focus_target,
                        0.2..=40.0,
                    )
                    .changed()
                {
                    if scene.camera.focus_smooth <= 1e-3 {
                        scene.camera.focus_distance = scene.camera.focus_target;
                    }
                    keep = true;
                }
                ui.label("F-stop");
                if ui
                    .slider("dof_fstop", &mut scene.camera.f_stop, 0.8..=22.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Focus range");
                if ui
                    .slider("dof_range", &mut post.dof.focus_range, 0.0..=2.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Max CoC");
                if ui
                    .slider("dof_coc", &mut post.dof.max_coc_px, 4.0..=48.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Scale");
                if ui
                    .slider("dof_scale", &mut post.dof.scale, 1.0..=40.0)
                    .changed()
                {
                    keep = true;
                }
                if ui
                    .checkbox("Half-res gather", &mut post.dof.half_res)
                    .changed()
                {
                    keep = true;
                }
                if ui.checkbox("Temporal", &mut post.dof.temporal).changed() {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("Tonemap", |ui| {
            if ui.checkbox("Enabled", &mut post.tonemap.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.tonemap.enabled, |ui| {
                if ui.checkbox("ACES", &mut post.tonemap.aces).changed() {
                    keep = true;
                }
                ui.label("Exposure");
                if ui
                    .slider("tm_exp", &mut post.tonemap.exposure, 0.1..=4.0)
                    .changed()
                {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("Color Grade", |ui| {
            if ui.checkbox("Enabled", &mut post.color_grade.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.color_grade.enabled, |ui| {
                ui.label("Contrast");
                if ui
                    .slider("cg_contrast", &mut post.color_grade.contrast, 0.0..=2.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Saturation");
                if ui
                    .slider("cg_sat", &mut post.color_grade.saturation, 0.0..=2.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Brightness");
                if ui
                    .slider("cg_bright", &mut post.color_grade.brightness, -0.5..=0.5)
                    .changed()
                {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("Vignette", |ui| {
            if ui.checkbox("Enabled", &mut post.vignette.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.vignette.enabled, |ui| {
                ui.label("Intensity");
                if ui
                    .slider("vig_int", &mut post.vignette.intensity, 0.0..=1.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Smoothness");
                if ui
                    .slider("vig_smooth", &mut post.vignette.smoothness, 0.05..=1.5)
                    .changed()
                {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("Film Grain", |ui| {
            if ui.checkbox("Enabled", &mut post.grain.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.grain.enabled, |ui| {
                ui.label("Intensity");
                if ui
                    .slider("grain_int", &mut post.grain.intensity, 0.0..=0.2)
                    .changed()
                {
                    keep = true;
                }
            });
        });

        ui.collapsing_header("FXAA", |ui| {
            if ui.checkbox("Enabled", &mut post.fxaa.enabled).changed() {
                keep = true;
            }
        });

        ui.collapsing_header("Fog", |ui| {
            if ui.checkbox("Enabled", &mut post.fog.enabled).changed() {
                keep = true;
            }
            ui.add_enabled(post.fog.enabled, |ui| {
                ui.label("Color");
                let mut fog_col = [
                    post.fog.color[0],
                    post.fog.color[1],
                    post.fog.color[2],
                    1.0,
                ];
                if ui.color_edit("fog_color", &mut fog_col).changed() {
                    post.fog.color = [fog_col[0], fog_col[1], fog_col[2]];
                    keep = true;
                }
                ui.label("Density");
                if ui
                    .slider("fog_dens", &mut post.fog.density, 0.0..=0.2)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Height");
                if ui
                    .slider("fog_height", &mut post.fog.height, -5.0..=5.0)
                    .changed()
                {
                    keep = true;
                }
                ui.label("Height falloff");
                if ui
                    .slider("fog_falloff", &mut post.fog.height_falloff, 0.0..=2.0)
                    .changed()
                {
                    keep = true;
                }
            });
        });
    });
    keep
}
