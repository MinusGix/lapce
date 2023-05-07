use std::sync::{atomic::AtomicU64, Arc};

use floem::{
    event::{Event, EventListner},
    peniko::kurbo::{Point, Rect, Size},
    reactive::{
        create_memo, ReadSignal, RwSignal, SignalGet, SignalGetUntracked, SignalSet,
        SignalWith, SignalWithUntracked,
    },
    style::{CursorStyle, Dimension, Style},
    view::View,
    views::{
        click, clip, container, label, list, rich_text, scroll, stack, svg,
        virtual_list, Decorators, VirtualListDirection, VirtualListItemSize,
    },
    AppContext,
};
use lapce_core::{
    cursor::{ColPosition, Cursor, CursorMode},
    mode::{Mode, VisualMode},
    selection::Selection,
};

use crate::{
    config::{color::LapceColor, icon::LapceIcons, LapceConfig},
    doc::{
        display_line::{VisualLine, VisualLineEntry},
        DocLine, Document, LineExtraStyle,
    },
    wave::wave_line,
    window_tab::Focus,
    workspace::LapceWorkspace,
};

use super::EditorData;

#[derive(Clone, Debug)]
enum CursorRender {
    CurrentLine {
        line_info: VisualLineEntry,
    },
    Selection {
        x: f64,
        width: f64,
        line_info: VisualLineEntry,
    },
    Caret {
        x: f64,
        width: f64,
        line_info: VisualLineEntry,
    },
}

fn cursor_caret(doc: &Document, offset: usize, block: bool) -> CursorRender {
    // let (line, col) = doc.buffer().offset_to_line_col(offset);
    let (line, col) = doc.visual_line_col_of_offset(offset);
    // TODO: don't unwrap
    let line_info = doc.visual_line_entry(line).unwrap();
    let phantom_text = doc.line_phantom_text2(line_info);
    let col = phantom_text.col_after(col, block);
    let x0 = doc.line_point_of_line_col(line_info, col, 12).x;
    if block {
        let right_offset = doc.buffer().move_right(offset, Mode::Insert, 1);
        // let (_, right_col) = doc.buffer().offset_to_line_col(right_offset);
        let (_, right_col) = doc.visual_line_col_of_offset(right_offset);
        let x1 = doc.line_point_of_line_col(line_info, right_col, 12).x;

        let width = if x1 > x0 { x1 - x0 } else { 7.0 };
        CursorRender::Caret {
            x: x0,
            width,
            line_info,
        }
    } else {
        CursorRender::Caret {
            x: x0 - 1.0,
            width: 2.0,
            line_info,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn visual_cursor(
    doc: &Document,
    start: usize,
    end: usize,
    mode: &VisualMode,
    horiz: Option<&ColPosition>,
    min_line: VisualLine,
    max_line: VisualLine,
    char_width: f64,
    is_active: bool,
) -> Vec<CursorRender> {
    // let (start_line, start_col) = doc.buffer().offset_to_line_col(start.min(end));
    // let (end_line, end_col) = doc.buffer().offset_to_line_col(start.max(end));
    // let (cursor_line, _) = doc.buffer().offset_to_line_col(end);
    let (start_line, start_col) = doc.visual_line_col_of_offset(start.min(end));
    let (end_line, end_col) = doc.visual_line_col_of_offset(start.max(end));
    let cursor_line = doc.visual_line_of_offset(end);

    let mut renders = Vec::new();

    // for line in min_line..max_line + 1 {
    for line_info in doc.iter_visual_lines(min_line) {
        let line = line_info.vline;
        if line > max_line {
            break;
        }

        if line < start_line {
            continue;
        }

        if line > end_line {
            break;
        }

        let left_col = match mode {
            VisualMode::Normal => {
                if start_line == line {
                    start_col
                } else {
                    0
                }
            }
            VisualMode::Linewise => 0,
            VisualMode::Blockwise => {
                let max_col = line_info.last_col(false);
                let left = start_col.min(end_col);
                if left > max_col {
                    continue;
                }
                left
            }
        };

        let (right_col, line_end) = match mode {
            VisualMode::Normal => {
                if line == end_line {
                    let max_col = line_info.last_col(true);

                    let end_offset =
                        doc.buffer().move_right(start.max(end), Mode::Visual, 1);
                    let (_, end_col) = doc.visual_line_col_of_offset(end_offset);

                    (end_col.min(max_col), false)
                } else {
                    (line_info.last_col(true), true)
                }
            }
            VisualMode::Linewise => (line_info.last_col(true), true),
            VisualMode::Blockwise => {
                let max_col = line_info.last_col(true);
                let right = match horiz.as_ref() {
                    Some(&ColPosition::End) => max_col,
                    _ => {
                        let end_offset =
                            doc.buffer().move_right(start.max(end), Mode::Visual, 1);
                        let (_, end_col) = doc.visual_line_col_of_offset(end_offset);
                        end_col.max(start_col).min(max_col)
                    }
                };
                (right, false)
            }
        };

        let phantom_text = doc.line_phantom_text2(line_info);
        let left_col = phantom_text.col_after(left_col, false);
        let right_col = phantom_text.col_after(right_col, false);
        let x0 = doc.line_point_of_line_col(line_info, left_col, 12).x;
        let mut x1 = doc.line_point_of_line_col(line_info, right_col, 12).x;
        if line_end {
            x1 += char_width;
        }

        renders.push(CursorRender::Selection {
            x: x0,
            width: x1 - x0,
            line_info,
        });

        if is_active && line == cursor_line {
            let caret = cursor_caret(doc, end, true);
            renders.push(caret);
        }
    }

    renders
}

fn insert_cursor(
    doc: &Document,
    selection: &Selection,
    min_line: VisualLine,
    max_line: VisualLine,
    char_width: f64,
    is_active: bool,
) -> Vec<CursorRender> {
    // let start = doc.buffer().offset_of_line(min_line);
    // let end = doc.buffer().offset_of_line(max_line + 1);
    let start = doc.offset_of_visual_line(min_line);
    let end = doc.offset_of_visual_line(VisualLine(max_line.get() + 1));
    let regions = selection.regions_in_range(start, end);

    let mut renders = Vec::new();

    for region in regions {
        let cursor_offset = region.end;
        // let (cursor_line, _) = doc.buffer().offset_to_line_col(cursor_offset);
        let cursor_line = doc.visual_line_of_offset(cursor_offset);
        let start = region.start;
        let end = region.end;
        // let (start_line, start_col) =
        //     doc.buffer().offset_to_line_col(start.min(end));
        // let (end_line, end_col) = doc.buffer().offset_to_line_col(start.max(end));
        let (start_line, start_col) = doc.visual_line_col_of_offset(start.min(end));
        let (end_line, end_col) = doc.visual_line_col_of_offset(start.max(end));
        for line_info in doc.iter_visual_lines(min_line) {
            let line = line_info.vline;
            if line > max_line {
                break;
            }

            if line < start_line {
                continue;
            }

            if line > end_line {
                break;
            }

            let left_col = match line {
                _ if line == start_line => start_col,
                _ => 0,
            };
            let (right_col, line_end) = match line {
                _ if line == end_line => {
                    let max_col = line_info.last_col(true);
                    (end_col.min(max_col), false)
                }
                // _ => (doc.buffer().line_end_col(line, true), true),
                _ => (line_info.last_col(true), true),
            };

            // Shift it by the inlay hints
            let phantom_text = doc.line_phantom_text2(line_info);
            let left_col = phantom_text.col_after(left_col, false);
            let right_col = phantom_text.col_after(right_col, false);

            let x0 = doc.line_point_of_line_col(line_info, left_col, 12).x;
            let mut x1 = doc.line_point_of_line_col(line_info, right_col, 12).x;
            if line_end {
                x1 += char_width;
            }

            if line == cursor_line {
                renders.push(CursorRender::CurrentLine { line_info });
            }

            if start != end {
                renders.push(CursorRender::Selection {
                    x: x0,
                    width: x1 - x0,
                    line_info,
                });
            }

            if is_active && line == cursor_line {
                let caret = cursor_caret(doc, cursor_offset, false);
                renders.push(caret);
            }
        }
    }
    renders
}

pub fn editor_view(
    cx: AppContext,
    workspace: Arc<LapceWorkspace>,
    is_active: impl Fn() -> bool + 'static + Copy,
    editor: RwSignal<EditorData>,
) -> impl View {
    let (cursor, viewport, config) = editor.with_untracked(|editor| {
        (
            editor.cursor.read_only(),
            editor.viewport,
            editor.common.config,
        )
    });

    stack(cx, move |cx| {
        (
            editor_breadcrumbs(cx, workspace, editor, config),
            container(cx, |cx| {
                stack(cx, |cx| {
                    (
                        editor_gutter(cx, editor, is_active),
                        stack(cx, move |cx| {
                            (
                                editor_cursor(
                                    cx,
                                    editor,
                                    cursor,
                                    viewport.read_only(),
                                    is_active,
                                    config,
                                ),
                                editor_content(cx, editor),
                            )
                        })
                        .style(cx, || Style::BASE.dimension_pct(1.0, 1.0)),
                    )
                })
                .style(cx, || Style::BASE.absolute().dimension_pct(1.0, 1.0))
            })
            .style(cx, || Style::BASE.dimension_pct(1.0, 1.0)),
        )
    })
    .style(cx, || Style::BASE.flex_col().dimension_pct(1.0, 1.0))
}

fn editor_gutter(
    cx: AppContext,
    editor: RwSignal<EditorData>,
    is_active: impl Fn() -> bool + 'static + Copy,
) -> impl View {
    let (cursor, viewport, scroll_delta, config) = editor.with(|editor| {
        (
            editor.cursor,
            editor.viewport,
            editor.scroll_delta,
            editor.common.config,
        )
    });

    let padding_left = 10.0;
    let padding_right = 30.0;

    let code_action_line = create_memo(cx.scope, move |_| {
        if is_active() {
            let doc = editor.with(|editor| editor.doc);
            let offset = cursor.with(|cursor| cursor.offset());
            doc.with(|doc| {
                let line = doc.buffer().line_of_offset(offset);
                let has_code_actions = doc
                    .code_actions
                    .get(&offset)
                    .map(|c| !c.1.is_empty())
                    .unwrap_or(false);
                if has_code_actions {
                    Some(line)
                } else {
                    None
                }
            })
        } else {
            None
        }
    });

    let current_line = create_memo(cx.scope, move |_| {
        let doc = editor.with(|editor| editor.doc);
        let (offset, mode) =
            cursor.with(|cursor| (cursor.offset(), cursor.get_mode()));
        let line = doc.with(|doc| {
            let line = doc.buffer().line_of_offset(offset);
            line
        });
        (line, mode)
    });

    stack(cx, |cx| {
        (
            stack(cx, |cx| {
                (
                    label(cx, || "".to_string())
                        .style(cx, move || Style::BASE.width_px(padding_left)),
                    label(cx, move || {
                        editor
                            .get()
                            .doc
                            .with(|doc| (doc.buffer().last_line() + 1).to_string())
                    }),
                    label(cx, || "".to_string())
                        .style(cx, move || Style::BASE.width_px(padding_right)),
                )
            })
            .style(cx, || Style::BASE.height_pct(1.0)),
            scroll(cx, |cx| {
                virtual_list(
                    cx,
                    VirtualListDirection::Vertical,
                    move || {
                        let editor = editor.get();
                        current_line.get();
                        editor.doc.get()
                    },
                    move |line: &DocLine| (line.line, current_line.get_untracked()),
                    move |cx, line: DocLine| {
                        let line_number = {
                            let config = config.get_untracked();
                            let (current_line, mode) = current_line.get_untracked();
                            if config.core.modal
                                && config.editor.modal_mode_relative_line_numbers
                                && mode != Mode::Insert
                            {
                                if line.line == current_line {
                                    line.line + 1
                                } else {
                                    line.line.abs_diff(current_line)
                                }
                            } else {
                                line.line + 1
                            }
                        };

                        stack(cx, move |cx| {
                            (
                                label(cx, || "".to_string()).style(cx, move || {
                                    Style::BASE.width_px(padding_left)
                                }),
                                label(cx, move || line_number.to_string()).style(
                                    cx,
                                    move || {
                                        let config = config.get();
                                        let (current_line, _) =
                                            current_line.get_untracked();
                                        Style::BASE
                                            .flex_grow(1.0)
                                            .apply_if(
                                                current_line != line.line,
                                                move |s| {
                                                    s.color(*config.get_color(
                                                        LapceColor::EDITOR_DIM,
                                                    ))
                                                },
                                            )
                                            .justify_end()
                                    },
                                ),
                                container(cx, |cx| {
                                    container(cx, |cx| {
                                        click(
                                            cx,
                                            |cx| {
                                                svg(cx, move || {
                                                    config.get().ui_svg(
                                                        LapceIcons::LIGHTBULB,
                                                    )
                                                })
                                                .style(cx, move || {
                                                    let config = config.get();
                                                    let size =
                                                        config.ui.icon_size() as f32;
                                                    Style::BASE
                                                        .dimension_px(size, size)
                                                        .color(*config.get_color(
                                                            LapceColor::LAPCE_WARN,
                                                        ))
                                                })
                                            },
                                            move || {
                                                editor.with_untracked(|editor| {
                                                    editor.show_code_actions(true);
                                                });
                                            },
                                        )
                                        .style(cx, move || {
                                            Style::BASE.apply_if(
                                                code_action_line.get()
                                                    != Some(line.line),
                                                |s| s.hide(),
                                            )
                                        })
                                    })
                                    .style(
                                        cx,
                                        move || {
                                            Style::BASE
                                                .justify_center()
                                                .items_center()
                                                .width_px(
                                                    padding_right - padding_left,
                                                )
                                        },
                                    )
                                })
                                .style(
                                    cx,
                                    move || {
                                        Style::BASE
                                            .justify_end()
                                            .width_px(padding_right)
                                    },
                                ),
                            )
                        })
                        .style(cx, move || {
                            let config = config.get_untracked();
                            let line_height = config.editor.line_height();
                            Style::BASE.items_center().height_px(line_height as f32)
                        })
                    },
                    VirtualListItemSize::Fixed(
                        config.get_untracked().editor.line_height() as f64,
                    ),
                )
                .style(cx, move || {
                    let config = config.get();
                    let padding_bottom = if config.editor.scroll_beyond_last_line {
                        viewport.get().height() as f32
                            - config.editor.line_height() as f32
                    } else {
                        0.0
                    };
                    Style::BASE
                        .flex_col()
                        .width_pct(1.0)
                        .padding_bottom(padding_bottom)
                })
            })
            .hide_bar(cx, || true)
            .on_event(EventListner::MouseWheel, move |event| {
                if let Event::MouseWheel(wheel) = event {
                    scroll_delta.set(wheel.wheel_delta);
                }
                true
            })
            .on_scroll_to(cx, move || {
                let viewport = viewport.get();
                Some(viewport.origin())
            })
            .style(cx, move || {
                Style::BASE
                    .absolute()
                    .background(
                        *config.get().get_color(LapceColor::EDITOR_BACKGROUND),
                    )
                    .dimension_pct(1.0, 1.0)
            }),
        )
    })
    .style(cx, move || {
        let config = config.get();
        Style::BASE
            .font_family(config.editor.font_family.clone())
            .font_size(config.editor.font_size as f32)
    })
}

fn editor_cursor(
    cx: AppContext,
    editor: RwSignal<EditorData>,
    cursor: ReadSignal<Cursor>,
    viewport: ReadSignal<Rect>,
    is_active: impl Fn() -> bool + 'static,
    config: ReadSignal<Arc<LapceConfig>>,
) -> impl View {
    let cursor = move || {
        let viewport = viewport.get();
        let config = config.get();
        let line_height = config.editor.line_height() as f64;

        let min_line = VisualLine((viewport.y0 / line_height).floor() as usize);
        let max_line = VisualLine((viewport.y1 / line_height).ceil() as usize);

        let is_active = is_active();
        let doc = editor.get().doc;
        doc.with(|doc| {
            cursor.with(|cursor| match &cursor.mode {
                CursorMode::Normal(offset) => {
                    // let line = doc.buffer().line_of_offset(*offset);
                    let line = doc.visual_line_of_offset(*offset);
                    // TODO: don't unwrap?
                    let line_info = doc.visual_line_entry(line).unwrap();
                    let mut renders =
                        vec![(viewport, CursorRender::CurrentLine { line_info })];
                    if is_active {
                        let caret = cursor_caret(doc, *offset, true);
                        renders.push((viewport, caret));
                    }
                    renders
                }
                CursorMode::Visual { start, end, mode } => visual_cursor(
                    doc, *start, *end, mode, None, min_line, max_line, 7.5,
                    is_active,
                )
                .into_iter()
                .map(|render| (viewport, render))
                .collect(),
                CursorMode::Insert(selection) => {
                    insert_cursor(doc, selection, min_line, max_line, 7.5, is_active)
                        .into_iter()
                        .map(|render| (viewport, render))
                        .collect()
                }
            })
        })
    };
    let id = AtomicU64::new(0);
    clip(cx, |cx| {
        list(
            cx,
            cursor,
            move |(_viewport, _cursor)| {
                id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            move |cx, (viewport, cursor)| {
                label(cx, || "".to_string()).style(cx, move || {
                    let config = config.get_untracked();
                    let line_height = config.editor.line_height();

                    let (width, margin_left, line_info, background) = match &cursor {
                        CursorRender::CurrentLine { line_info } => (
                            Dimension::Percent(1.0),
                            0.0,
                            *line_info,
                            *config.get_color(LapceColor::EDITOR_CURRENT_LINE),
                        ),
                        CursorRender::Selection {
                            x,
                            width,
                            line_info,
                        } => (
                            Dimension::Points(*width as f32),
                            (*x - viewport.x0) as f32,
                            *line_info,
                            *config.get_color(LapceColor::EDITOR_SELECTION),
                        ),
                        CursorRender::Caret {
                            x,
                            width,
                            line_info,
                        } => (
                            Dimension::Points(*width as f32),
                            (*x - viewport.x0) as f32,
                            *line_info,
                            *config.get_color(LapceColor::EDITOR_CARET),
                        ),
                    };

                    Style::BASE
                        .absolute()
                        .width(width)
                        .height_px(line_height as f32)
                        .margin_left(margin_left)
                        .margin_top(
                            (line_info.vline.get() * line_height) as f32
                                - viewport.y0 as f32,
                        )
                        .background(background)
                })
            },
        )
        .style(cx, move || Style::BASE.dimension_pct(1.0, 1.0))
    })
    .style(cx, move || Style::BASE.absolute().dimension_pct(1.0, 1.0))
}

fn editor_breadcrumbs(
    cx: AppContext,
    workspace: Arc<LapceWorkspace>,
    editor: RwSignal<EditorData>,
    config: ReadSignal<Arc<LapceConfig>>,
) -> impl View {
    stack(cx, move |cx| {
        (
            label(cx, || " ".to_string()).style(cx, || Style::BASE.margin_vert(5.0)),
            scroll(cx, move |cx| {
                let workspace = workspace.clone();
                list(
                    cx,
                    move || {
                        let doc = editor.with(|editor| editor.doc);
                        let full_path = doc
                            .with_untracked(|doc| doc.content.path().cloned())
                            .unwrap_or_default();
                        let mut path = full_path.clone();
                        if let Some(workspace_path) = workspace.clone().path.as_ref()
                        {
                            path = path
                                .strip_prefix(workspace_path)
                                .unwrap_or(&full_path)
                                .to_path_buf();
                        }
                        path.ancestors()
                            .collect::<Vec<_>>()
                            .iter()
                            .rev()
                            .filter_map(|path| {
                                Some(path.file_name()?.to_str()?.to_string())
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .enumerate()
                    },
                    |(i, section)| (*i, section.to_string()),
                    move |cx, (i, section)| {
                        stack(cx, move |cx| {
                            (
                                svg(cx, move || {
                                    config
                                        .get()
                                        .ui_svg(LapceIcons::BREADCRUMB_SEPARATOR)
                                })
                                .style(
                                    cx,
                                    move || {
                                        let config = config.get();
                                        let size = config.ui.icon_size() as f32;
                                        Style::BASE
                                            .apply_if(i == 0, |s| s.hide())
                                            .dimension_px(size, size)
                                            .color(*config.get_color(
                                                LapceColor::LAPCE_ICON_ACTIVE,
                                            ))
                                    },
                                ),
                                label(cx, move || section.clone()),
                            )
                        })
                        .style(cx, || Style::BASE.items_center())
                    },
                )
                .style(cx, || Style::BASE.padding_horiz(10.0))
            })
            .on_scroll_to(cx, move || {
                editor.with(|_editor| ());
                Some(Point::new(3000.0, 0.0))
            })
            .hide_bar(cx, || true)
            .style(cx, move || {
                Style::BASE
                    .absolute()
                    .dimension_pct(1.0, 1.0)
                    .border_bottom(1.0)
                    .border_color(*config.get().get_color(LapceColor::LAPCE_BORDER))
                    .items_center()
            }),
        )
    })
    .style(cx, move || {
        let config = config.get_untracked();
        let line_height = config.editor.line_height();
        Style::BASE.items_center().height_px(line_height as f32)
    })
}

fn editor_content(cx: AppContext, editor: RwSignal<EditorData>) -> impl View {
    let (cursor, scroll_delta, scroll_to, window_origin, viewport, config) = editor
        .with_untracked(|editor| {
            (
                editor.cursor.read_only(),
                editor.scroll_delta.read_only(),
                editor.scroll_to,
                editor.window_origin,
                editor.viewport,
                editor.common.config,
            )
        });

    let key_fn = move |line: &DocLine| {
        (
            editor
                .with_untracked(|editor| editor.doc)
                .with_untracked(|doc| doc.content.clone()),
            line.rev,
            line.style_rev,
            line.line,
        )
    };
    let view_fn = move |cx, line: DocLine| {
        let extra_styles = line.text.extra_style.clone();
        stack(cx, |cx| {
            (
                editor_extra_style(cx, extra_styles, config),
                rich_text(cx, move || line.text.clone().text.clone()),
            )
        })
        .style(cx, move || {
            let config = config.get_untracked();
            let line_height = config.editor.line_height();
            Style::BASE.items_center().height_px(line_height as f32)
        })
    };

    scroll(cx, |cx| {
        let focus = editor.with_untracked(|e| e.common.focus);
        click(
            cx,
            |cx| {
                let line_height = config.get_untracked().editor.line_height();
                virtual_list(
                    cx,
                    VirtualListDirection::Vertical,
                    move || editor.get().doc.get(),
                    key_fn,
                    view_fn,
                    VirtualListItemSize::Fixed(line_height as f64),
                )
                .style(cx, move || {
                    let config = config.get();
                    let padding_bottom = if config.editor.scroll_beyond_last_line {
                        viewport.get().height() as f32
                            - config.editor.line_height() as f32
                    } else {
                        0.0
                    };
                    Style::BASE
                        .flex_col()
                        .padding_bottom(padding_bottom)
                        .cursor(CursorStyle::Text)
                        .min_width_pct(1.0)
                })
            },
            move || {
                focus.set(Focus::Workbench);
            },
        )
        .style(cx, || Style::BASE.min_dimension_pct(1.0, 1.0))
    })
    .scroll_bar_color(cx, move || {
        *config.get().get_color(LapceColor::LAPCE_SCROLL_BAR)
    })
    .on_resize(move |point, _rect| {
        window_origin.set(point);
    })
    .onscroll(move |rect| {
        viewport.set(rect);
    })
    .on_scroll_to(cx, move || scroll_to.get().map(|s| s.to_point()))
    .on_scroll_delta(cx, move || scroll_delta.get())
    .on_ensure_visible(cx, move || {
        let cursor = cursor.get();
        let offset = cursor.offset();
        let editor = editor.get_untracked();
        let doc = editor.doc;
        let caret =
            doc.with_untracked(|doc| cursor_caret(doc, offset, !cursor.is_insert()));
        let config = config.get_untracked();
        let line_height = config.editor.line_height();
        if let CursorRender::Caret {
            x,
            width,
            line_info,
        } = caret
        {
            let rect = Size::new(width, line_height as f64).to_rect().with_origin(
                Point::new(x, (line_info.vline.get() * line_height) as f64),
            );

            let viewport = viewport.get_untracked();
            let smallest_distance = (viewport.y0 - rect.y0)
                .abs()
                .min((viewport.y1 - rect.y0).abs())
                .min((viewport.y0 - rect.y1).abs())
                .min((viewport.y1 - rect.y1).abs());
            let biggest_distance = (viewport.y0 - rect.y0)
                .abs()
                .max((viewport.y1 - rect.y0).abs())
                .max((viewport.y0 - rect.y1).abs())
                .max((viewport.y1 - rect.y1).abs());
            let jump_to_middle = biggest_distance > viewport.height()
                && smallest_distance > viewport.height() / 2.0;

            if jump_to_middle {
                rect.inflate(0.0, viewport.height() / 2.0)
            } else {
                rect.inflate(
                    0.0,
                    (config.editor.cursor_surrounding_lines * line_height) as f64,
                )
            }
        } else {
            Rect::ZERO
        }
    })
    .style(cx, || Style::BASE.absolute().dimension_pct(1.0, 1.0))
}

fn editor_extra_style(
    cx: AppContext,
    extra_styles: Vec<LineExtraStyle>,
    config: ReadSignal<Arc<LapceConfig>>,
) -> impl View {
    list(
        cx,
        move || extra_styles.clone(),
        |_| 0,
        move |cx, extra| {
            container(cx, |cx| {
                stack(cx, |cx| {
                    (
                        label(cx, || " ".to_string()).style(cx, move || {
                            let config = config.get();
                            Style::BASE
                                .font_family(config.editor.font_family.clone())
                                .font_size(config.editor.font_size as f32)
                                .width_pct(1.0)
                                .apply_opt(extra.bg_color, Style::background)
                        }),
                        wave_line(cx).style(cx, move || {
                            Style::BASE
                                .absolute()
                                .dimension_pct(1.0, 1.0)
                                .apply_opt(extra.wave_line, Style::color)
                        }),
                    )
                })
                .style(cx, || Style::BASE.width_pct(1.0))
            })
            .style(cx, move || {
                let line_height = config.get().editor.line_height();
                Style::BASE
                    .absolute()
                    .height_px(line_height as f32)
                    .width(match extra.width {
                        Some(width) => Dimension::Points(width as f32),
                        None => Dimension::Percent(1.0),
                    })
                    .apply_if(extra.width.is_some(), |s| {
                        s.margin_left(extra.x as f32)
                    })
                    .items_center()
            })
        },
    )
    .style(cx, || Style::BASE.absolute().dimension_pct(1.0, 1.0))
}
