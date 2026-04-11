//! Live top-K next-token prediction panel.
//!
//! Fixed UI element in the top-right corner. At startup we spawn a stack of
//! text rows ("1. …", "2. …", …). A per-frame system reads the latest
//! `LogitsResult` from `GptState` and rewrites the row text with the
//! predicted token string and its logit.

use bevy::prelude::*;

use crate::model::GptState;

pub struct PredictionPlugin;

impl Plugin for PredictionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_prediction_ui)
            .add_systems(Update, update_prediction_ui);
    }
}

#[derive(Component)]
pub struct PredictionRow(pub usize);

#[derive(Component)]
pub struct PredictionHeader;

const TOP_K: usize = 7;

fn spawn_prediction_ui(mut commands: Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(20.0),
                right: Val::Px(20.0),
                width: Val::Px(330.0),
                padding: UiRect::all(Val::Px(16.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                ..default()
            },
            background_color: BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            ..default()
        })
        .with_children(|panel| {
            panel.spawn((
                TextBundle::from_section(
                    "NEXT TOKEN  ·  computing…",
                    TextStyle {
                        font_size: 18.0,
                        color: Color::srgb(0.9, 0.95, 1.0),
                        ..default()
                    },
                ),
                PredictionHeader,
            ));
            for i in 0..TOP_K {
                panel.spawn((
                    TextBundle::from_section(
                        format!("{}. …", i + 1),
                        TextStyle {
                            font_size: 16.0,
                            color: Color::srgb(0.72, 0.82, 0.92),
                            ..default()
                        },
                    ),
                    PredictionRow(i),
                ));
            }
        });
}

fn update_prediction_ui(
    state: NonSend<GptState>,
    mut rows: Query<(&PredictionRow, &mut Text), Without<PredictionHeader>>,
    mut header: Query<&mut Text, With<PredictionHeader>>,
) {
    let Some(logits) = &state.logits else { return };
    let preds = logits.top_k_predictions(&state.model.config, TOP_K);

    if let Ok(mut h) = header.get_single_mut() {
        let status = if state.computing {
            "streaming…"
        } else {
            "ready"
        };
        h.sections[0].value = format!("NEXT TOKEN  ·  {status}");
    }

    for (row, mut text) in rows.iter_mut() {
        if let Some(&(tok_id, score)) = preds.get(row.0) {
            let tok = state
                .tokenizer
                .id_to_token(tok_id as u32)
                .unwrap_or_else(|| format!("[{tok_id}]"));
            let clean = tok.replace("Ġ", " ").replace("Ċ", "\\n");
            // Pad the token slot so logits line up in a column.
            let padded = format!("{:<14}", clean);
            text.sections[0].value = format!("{}. {padded}{score:>6.1}", row.0 + 1);

            // Highlight the top prediction in bright yellow-green.
            let c = if row.0 == 0 {
                Color::srgb(0.55, 1.0, 0.55)
            } else {
                let dim = 0.95 - (row.0 as f32) * 0.1;
                Color::srgb(dim * 0.75, dim * 0.85, dim)
            };
            text.sections[0].style.color = c;
        } else {
            text.sections[0].value = format!("{}. …", row.0 + 1);
        }
    }
}
