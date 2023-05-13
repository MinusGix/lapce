use std::{cmp::Ordering, str::FromStr, sync::Arc};

use anyhow::Result;
use floem::{
    ext_event::create_ext_action,
    glazier::Modifiers,
    peniko::kurbo::{Point, Rect, Vec2},
    reactive::{
        create_rw_signal, use_context, RwSignal, Scope, SignalGetUntracked,
        SignalSet, SignalUpdate, SignalWithUntracked,
    },
};
use lapce_core::{
    buffer::InvalLines,
    command::{EditCommand, FocusCommand},
    cursor::{Cursor, CursorAffinity, CursorMode},
    editor::EditType,
    mode::Mode,
    movement::Movement,
    selection::{InsertDrift, Selection},
    syntax::edit::SyntaxEdit,
};
use lapce_rpc::{plugin::PluginId, proxy::ProxyResponse};
use lapce_xi_rope::{Rope, RopeDelta, Transformer};
use lsp_types::{
    CompletionItem, CompletionTextEdit, GotoDefinitionResponse, Location, TextEdit,
};
use serde::{Deserialize, Serialize};

use crate::{
    command::{CommandExecuted, InternalCommand},
    completion::CompletionStatus,
    db::LapceDb,
    doc::{DocContent, DocWrap, Document},
    editor::location::{EditorLocation, EditorPosition},
    editor_tab::EditorTabChild,
    find::Find,
    id::{EditorId, EditorTabId},
    keypress::{condition::Condition, KeyPressFocus},
    main_split::{MainSplitData, SplitDirection, SplitMoveDirection},
    proxy::path_from_url,
    snippet::Snippet,
    window_tab::{CommonData, WindowTabData},
};

pub mod location;
pub mod view;

#[derive(Clone, Debug)]
pub enum InlineFindDirection {
    Left,
    Right,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EditorInfo {
    pub content: DocContent,
    pub wrap: DocWrap,
    // pub unsaved: Option<String>,
    pub offset: usize,
    pub scroll_offset: (f64, f64),
}

impl EditorInfo {
    pub fn to_data(
        &self,
        cx: Scope,
        data: MainSplitData,
        editor_tab_id: EditorTabId,
    ) -> RwSignal<EditorData> {
        let editor_id = EditorId::next();
        let editor_data = match &self.content {
            DocContent::File(path) => {
                let (doc, new_doc) = data.get_doc(cx, path.clone());
                let editor_data = EditorData::new(
                    cx,
                    Some(editor_tab_id),
                    editor_id,
                    doc,
                    data.common,
                );
                editor_data.go_to_location(
                    cx,
                    EditorLocation {
                        path: path.clone(),
                        position: Some(EditorPosition::Offset(self.offset)),
                        scroll_offset: Some(Vec2::new(
                            self.scroll_offset.0,
                            self.scroll_offset.1,
                        )),
                        ignore_unconfirmed: false,
                        same_editor_tab: false,
                    },
                    new_doc,
                    None,
                );
                editor_data
            }
            DocContent::Local => {
                EditorData::new_local(cx, editor_id, data.common, self.wrap)
            }
        };
        let editor_data = create_rw_signal(cx, editor_data);
        data.editors.update(|editors| {
            editors.insert(editor_id, editor_data);
        });
        editor_data
    }
}

pub type SnippetIndex = Vec<(usize, (usize, usize))>;

#[derive(Clone)]
pub struct EditorData {
    pub editor_tab_id: Option<EditorTabId>,
    pub editor_id: EditorId,
    pub doc: RwSignal<Document>,
    pub confirmed: RwSignal<bool>,
    pub cursor: RwSignal<Cursor>,
    pub window_origin: RwSignal<Point>,
    pub viewport: RwSignal<Rect>,
    pub scroll_delta: RwSignal<Vec2>,
    pub scroll_to: RwSignal<Option<Vec2>>,
    pub snippet: RwSignal<Option<SnippetIndex>>,
    pub find: RwSignal<Find>,
    pub last_movement: RwSignal<Movement>,
    pub inline_find: RwSignal<Option<InlineFindDirection>>,
    pub last_inline_find: RwSignal<Option<(InlineFindDirection, String)>>,
    pub common: CommonData,
}

impl PartialEq for EditorData {
    fn eq(&self, other: &Self) -> bool {
        self.editor_id == other.editor_id
    }
}

impl EditorData {
    pub fn new(
        cx: Scope,
        editor_tab_id: Option<EditorTabId>,
        editor_id: EditorId,
        doc: RwSignal<Document>,
        common: CommonData,
    ) -> Self {
        let modal = common.config.with_untracked(|c| c.core.modal);
        let cursor = Cursor::new(
            if modal {
                CursorMode::Normal(0)
            } else {
                CursorMode::Insert(Selection::caret(0))
            },
            None,
            None,
            // TODO: is this the right affinity?
            CursorAffinity::Backward,
        );
        let cursor = create_rw_signal(cx, cursor);
        let scroll_delta = create_rw_signal(cx, Vec2::ZERO);
        let scroll_to = create_rw_signal(cx, None);
        let snippet = create_rw_signal(cx, None);
        let window_origin = create_rw_signal(cx, Point::ZERO);
        let viewport = create_rw_signal(cx, Rect::ZERO);
        let confirmed = create_rw_signal(cx, false);
        let find = create_rw_signal(cx, Find::new(0));
        let last_movement = create_rw_signal(cx, Movement::Left);
        let inline_find = create_rw_signal(cx, None);
        let last_inline_find = create_rw_signal(cx, None);
        Self {
            editor_tab_id,
            editor_id,
            doc,
            cursor,
            confirmed,
            snippet,
            window_origin,
            viewport,
            scroll_delta,
            scroll_to,
            find,
            last_movement,
            inline_find,
            last_inline_find,
            common,
        }
    }

    pub fn new_local(
        cx: Scope,
        editor_id: EditorId,
        common: CommonData,
        wrap: DocWrap,
    ) -> Self {
        let doc = Document::new_local(cx, common.proxy.clone(), common.config, wrap);
        let doc = create_rw_signal(cx, doc);
        Self::new(cx, None, editor_id, doc, common)
    }

    pub fn editor_info(&self, _data: &WindowTabData) -> EditorInfo {
        // let unsaved = if let BufferContent::Scratch(id, _) = &self.content {
        //     let doc = data.main_split.scratch_docs.get(id).unwrap();
        //     Some(doc.buffer().to_string())
        // } else {
        //     None
        // };

        let offset = self.cursor.get_untracked().offset();
        let scroll_offset = self.viewport.get_untracked().origin();
        EditorInfo {
            content: self.doc.get_untracked().content,
            wrap: self.doc.get_untracked().wrap,
            offset,
            scroll_offset: (scroll_offset.x, scroll_offset.y),
        }
    }

    pub fn copy(
        &self,
        cx: Scope,
        editor_tab_id: Option<EditorTabId>,
        editor_id: EditorId,
    ) -> Self {
        let mut editor = self.clone();
        editor.cursor = create_rw_signal(cx, editor.cursor.get_untracked());
        editor.viewport = create_rw_signal(cx, editor.viewport.get_untracked());
        editor.scroll_delta = create_rw_signal(cx, Vec2::ZERO);
        editor.scroll_to = create_rw_signal(
            cx,
            Some(editor.viewport.get_untracked().origin().to_vec2()),
        );
        editor.window_origin = create_rw_signal(cx, Point::ZERO);
        editor.confirmed = create_rw_signal(cx, true);
        editor.snippet = create_rw_signal(cx, None);
        editor.editor_tab_id = editor_tab_id;
        editor.editor_id = editor_id;
        editor
    }

    fn run_edit_command(&self, _cx: Scope, cmd: &EditCommand) -> CommandExecuted {
        let modal = self
            .common
            .config
            .with_untracked(|config| config.core.modal)
            && !self.doc.with_untracked(|doc| doc.content.is_local());
        let doc_before_edit =
            self.doc.with_untracked(|doc| doc.buffer().text().clone());
        let mut cursor = self.cursor.get_untracked();
        let mut register = self.common.register.get_untracked();

        let yank_data =
            if let lapce_core::cursor::CursorMode::Visual { .. } = &cursor.mode {
                Some(self.doc.with_untracked(|doc| cursor.yank(doc.buffer())))
            } else {
                None
            };

        let deltas = self
            .doc
            .try_update(|doc| doc.do_edit(&mut cursor, cmd, modal, &mut register))
            .unwrap();

        if !deltas.is_empty() {
            if let Some(data) = yank_data {
                register.add_delete(data);
            }
        }

        self.cursor.set(cursor);
        self.common.register.set(register);

        if show_completion(cmd, &doc_before_edit, &deltas) {
            self.update_completion(false);
        } else {
            self.cancel_completion();
        }
        self.apply_deltas(&deltas);
        if let EditCommand::NormalMode = cmd {
            self.snippet.set(None);
        }

        CommandExecuted::Yes
    }

    fn run_move_command(
        &self,
        _cx: Scope,
        movement: &lapce_core::movement::Movement,
        count: Option<usize>,
        mods: Modifiers,
    ) -> CommandExecuted {
        if movement.is_jump() && movement != &self.last_movement.get_untracked() {
            let path = self.doc.with_untracked(|doc| doc.content.path().cloned());
            if let Some(path) = path {
                let offset = self.cursor.with_untracked(|c| c.offset());
                let scroll_offset = self.viewport.get_untracked().origin().to_vec2();
                self.common.internal_command.set(Some(
                    InternalCommand::SaveJumpLocation {
                        path,
                        offset,
                        scroll_offset,
                    },
                ));
            }
        }
        self.last_movement.set(movement.clone());

        let mut cursor = self.cursor.get_untracked();
        let config = self.common.config.get_untracked();
        self.doc.update(|doc| {
            self.common.register.update(|register| {
                doc.move_cursor(
                    &mut cursor,
                    movement,
                    count.unwrap_or(1),
                    mods.shift(),
                    register,
                    &config,
                );
            });
        });
        self.cursor.set(cursor);

        if self.snippet.with_untracked(|s| s.is_some()) {
            self.snippet.update(|snippet| {
                let offset = self.cursor.get_untracked().offset();
                let mut within_region = false;
                for (_, (start, end)) in snippet.as_mut().unwrap() {
                    if offset >= *start && offset <= *end {
                        within_region = true;
                        break;
                    }
                }
                if !within_region {
                    *snippet = None;
                }
            })
        }
        self.cancel_completion();
        CommandExecuted::Yes
    }

    fn run_focus_command(
        &self,
        cx: Scope,
        cmd: &FocusCommand,
        count: Option<usize>,
        mods: Modifiers,
    ) -> CommandExecuted {
        match cmd {
            FocusCommand::SplitVertical => {
                if let Some(editor_tab_id) = self.editor_tab_id {
                    self.common
                        .internal_command
                        .set(Some(InternalCommand::Split {
                            direction: SplitDirection::Vertical,
                            editor_tab_id,
                        }));
                }
            }
            FocusCommand::SplitHorizontal => {
                if let Some(editor_tab_id) = self.editor_tab_id {
                    self.common
                        .internal_command
                        .set(Some(InternalCommand::Split {
                            direction: SplitDirection::Horizontal,
                            editor_tab_id,
                        }));
                }
            }
            FocusCommand::SplitRight => {
                if let Some(editor_tab_id) = self.editor_tab_id {
                    self.common.internal_command.set(Some(
                        InternalCommand::SplitMove {
                            direction: SplitMoveDirection::Right,
                            editor_tab_id,
                        },
                    ));
                }
            }
            FocusCommand::SplitLeft => {
                if let Some(editor_tab_id) = self.editor_tab_id {
                    self.common.internal_command.set(Some(
                        InternalCommand::SplitMove {
                            direction: SplitMoveDirection::Left,
                            editor_tab_id,
                        },
                    ));
                }
            }
            FocusCommand::SplitUp => {
                if let Some(editor_tab_id) = self.editor_tab_id {
                    self.common.internal_command.set(Some(
                        InternalCommand::SplitMove {
                            direction: SplitMoveDirection::Up,
                            editor_tab_id,
                        },
                    ));
                }
            }
            FocusCommand::SplitDown => {
                if let Some(editor_tab_id) = self.editor_tab_id {
                    self.common.internal_command.set(Some(
                        InternalCommand::SplitMove {
                            direction: SplitMoveDirection::Down,
                            editor_tab_id,
                        },
                    ));
                }
            }
            FocusCommand::SplitExchange => {
                if let Some(editor_tab_id) = self.editor_tab_id {
                    self.common
                        .internal_command
                        .set(Some(InternalCommand::SplitExchange { editor_tab_id }));
                }
            }
            FocusCommand::SplitClose => {
                if let Some(editor_tab_id) = self.editor_tab_id {
                    self.common.internal_command.set(Some(
                        InternalCommand::EditorTabChildClose {
                            editor_tab_id,
                            child: EditorTabChild::Editor(self.editor_id),
                        },
                    ));
                }
            }
            FocusCommand::PageUp => {
                self.page_move(cx, false, mods);
            }
            FocusCommand::PageDown => {
                self.page_move(cx, true, mods);
            }
            FocusCommand::ScrollUp => {
                self.scroll(cx, false, count.unwrap_or(1), mods);
            }
            FocusCommand::ScrollDown => {
                self.scroll(cx, true, count.unwrap_or(1), mods);
            }
            FocusCommand::ListNext => {
                self.common.completion.update(|c| {
                    c.next();
                });
            }
            FocusCommand::ListPrevious => {
                self.common.completion.update(|c| {
                    c.previous();
                });
            }
            FocusCommand::ListNextPage => {
                self.common.completion.update(|c| {
                    c.next_page();
                });
            }
            FocusCommand::ListPreviousPage => {
                self.common.completion.update(|c| {
                    c.previous_page();
                });
            }
            FocusCommand::ListSelect => {
                self.select_completion(cx);
            }
            FocusCommand::JumpToNextSnippetPlaceholder => {
                self.snippet.update(|snippet| {
                    if let Some(snippet_mut) = snippet.as_mut() {
                        let mut current = 0;
                        let offset = self.cursor.get_untracked().offset();
                        for (i, (_, (start, end))) in snippet_mut.iter().enumerate()
                        {
                            if *start <= offset && offset <= *end {
                                current = i;
                                break;
                            }
                        }

                        let last_placeholder = current + 1 >= snippet_mut.len() - 1;

                        if let Some((_, (start, end))) = snippet_mut.get(current + 1)
                        {
                            let mut selection =
                                lapce_core::selection::Selection::new();
                            let region = lapce_core::selection::SelRegion::new(
                                *start, *end, None,
                            );
                            selection.add_region(region);
                            self.cursor.update(|cursor| {
                                cursor.set_insert(selection);
                            });
                        }

                        if last_placeholder {
                            *snippet = None;
                        }
                        // self.update_signature();
                        self.cancel_completion();
                    }
                });
            }
            FocusCommand::JumpToPrevSnippetPlaceholder => {
                self.snippet.update(|snippet| {
                    if let Some(snippet_mut) = snippet.as_mut() {
                        let mut current = 0;
                        let offset = self.cursor.get_untracked().offset();
                        for (i, (_, (start, end))) in snippet_mut.iter().enumerate()
                        {
                            if *start <= offset && offset <= *end {
                                current = i;
                                break;
                            }
                        }

                        if current > 0 {
                            if let Some((_, (start, end))) =
                                snippet_mut.get(current - 1)
                            {
                                let mut selection =
                                    lapce_core::selection::Selection::new();
                                let region = lapce_core::selection::SelRegion::new(
                                    *start, *end, None,
                                );
                                selection.add_region(region);
                                self.cursor.update(|cursor| {
                                    cursor.set_insert(selection);
                                });
                            }
                            // self.update_signature();
                            self.cancel_completion();
                        }
                    }
                });
            }
            FocusCommand::GotoDefinition => {
                self.go_to_definition(cx);
            }
            FocusCommand::ShowCodeActions => {
                self.show_code_actions(false);
            }
            FocusCommand::SearchWholeWordForward => {
                self.search_whole_word_forward(cx, mods);
            }
            FocusCommand::SearchForward => {
                self.search_forward(cx, mods);
            }
            FocusCommand::SearchBackward => {
                self.search_backward(cx, mods);
            }
            FocusCommand::Save => {
                self.save(cx, false, true);
            }
            FocusCommand::InlineFindLeft => {
                self.inline_find.set(Some(InlineFindDirection::Left));
            }
            FocusCommand::InlineFindRight => {
                self.inline_find.set(Some(InlineFindDirection::Right));
            }
            FocusCommand::RepeatLastInlineFind => {
                if let Some((direction, c)) = self.last_inline_find.get_untracked() {
                    self.inline_find(cx, direction, &c);
                }
            }
            _ => {}
        }
        CommandExecuted::Yes
    }

    /// Jump to the next/previous column on the line which matches the given text
    fn inline_find(&self, cx: Scope, direction: InlineFindDirection, c: &str) {
        let offset = self.cursor.with_untracked(|c| c.offset());
        let (line_content, line_start_offset) = self.doc.with_untracked(|doc| {
            let line = doc.buffer().line_of_offset(offset);
            let line_content = doc.buffer().line_content(line);
            let line_start_offset = doc.buffer().offset_of_line(line);
            (line_content.to_string(), line_start_offset)
        });
        let index = offset - line_start_offset;
        if let Some(new_index) = match direction {
            InlineFindDirection::Left => line_content[..index].rfind(c),
            InlineFindDirection::Right => {
                if index + 1 >= line_content.len() {
                    None
                } else {
                    let index = index
                        + self.doc.with_untracked(|doc| {
                            doc.buffer().next_grapheme_offset(
                                offset,
                                1,
                                doc.buffer().offset_line_end(offset, false),
                            )
                        })
                        - offset;
                    line_content[index..].find(c).map(|i| i + index)
                }
            }
        } {
            self.run_move_command(
                cx,
                &lapce_core::movement::Movement::Offset(
                    new_index + line_start_offset,
                ),
                None,
                Modifiers::empty(),
            );
        }
    }

    fn go_to_definition(&self, cx: Scope) {
        let path = match self.doc.with_untracked(|doc| {
            if doc.loaded() {
                doc.content.path().cloned()
            } else {
                None
            }
        }) {
            Some(path) => path,
            None => return,
        };

        let offset = self.cursor.with_untracked(|c| c.offset());
        let (start_position, position) = self.doc.with_untracked(|doc| {
            let start_offset = doc.buffer().prev_code_boundary(offset);
            let start_position = doc.buffer().offset_to_position(start_offset);
            let position = doc.buffer().offset_to_position(offset);
            (start_position, position)
        });

        enum DefinitionOrReferece {
            Location(EditorLocation),
            References(Vec<Location>),
        }

        let internal_command = self.common.internal_command;
        let cursor = self.cursor.read_only();
        let send = create_ext_action(cx, move |d| {
            let current_offset = cursor.with_untracked(|c| c.offset());
            if current_offset != offset {
                return;
            }

            match d {
                DefinitionOrReferece::Location(location) => {
                    internal_command
                        .set(Some(InternalCommand::JumpToLocation { location }));
                }
                DefinitionOrReferece::References(locations) => {
                    internal_command.set(Some(InternalCommand::PaletteReferences {
                        references: locations
                            .into_iter()
                            .map(|l| EditorLocation {
                                path: path_from_url(&l.uri),
                                position: Some(EditorPosition::Position(
                                    l.range.start,
                                )),
                                scroll_offset: None,
                                ignore_unconfirmed: false,
                                same_editor_tab: false,
                            })
                            .collect(),
                    }));
                }
            }
        });
        let proxy = self.common.proxy.clone();
        self.common.proxy.get_definition(
            offset,
            path.clone(),
            position,
            move |result| {
                if let Ok(ProxyResponse::GetDefinitionResponse {
                    definition, ..
                }) = result
                {
                    if let Some(location) = match definition {
                        GotoDefinitionResponse::Scalar(location) => Some(location),
                        GotoDefinitionResponse::Array(locations) => {
                            if !locations.is_empty() {
                                Some(locations[0].clone())
                            } else {
                                None
                            }
                        }
                        GotoDefinitionResponse::Link(location_links) => {
                            let location_link = location_links[0].clone();
                            Some(Location {
                                uri: location_link.target_uri,
                                range: location_link.target_selection_range,
                            })
                        }
                    } {
                        if location.range.start == start_position {
                            proxy.get_references(
                                path.clone(),
                                position,
                                move |result| {
                                    if let Ok(
                                        ProxyResponse::GetReferencesResponse {
                                            references,
                                        },
                                    ) = result
                                    {
                                        if references.is_empty() {
                                            return;
                                        }
                                        if references.len() == 1 {
                                            let location = &references[0];
                                            send(DefinitionOrReferece::Location(
                                                EditorLocation {
                                                    path: path_from_url(
                                                        &location.uri,
                                                    ),
                                                    position: Some(
                                                        EditorPosition::Position(
                                                            location.range.start,
                                                        ),
                                                    ),
                                                    scroll_offset: None,
                                                    ignore_unconfirmed: false,
                                                    same_editor_tab: false,
                                                },
                                            ));
                                        } else {
                                            send(DefinitionOrReferece::References(
                                                references,
                                            ));
                                        }
                                    }
                                },
                            );
                        } else {
                            let path = path_from_url(&location.uri);
                            send(DefinitionOrReferece::Location(EditorLocation {
                                path,
                                position: Some(EditorPosition::Position(
                                    location.range.start,
                                )),
                                scroll_offset: None,
                                ignore_unconfirmed: false,
                                same_editor_tab: false,
                            }));
                        }
                    }
                }
            },
        );
    }

    fn page_move(&self, cx: Scope, down: bool, mods: Modifiers) {
        let config = self.common.config.get_untracked();
        let viewport = self.viewport.get_untracked();
        let line_height = config.editor.line_height() as f64;
        let lines = (viewport.height() / line_height / 2.0).round() as usize;
        let distance = (lines as f64) * line_height;
        self.scroll_delta
            .set(Vec2::new(0.0, if down { distance } else { -distance }));
        self.run_move_command(
            cx,
            if down {
                &lapce_core::movement::Movement::Down
            } else {
                &lapce_core::movement::Movement::Up
            },
            Some(lines),
            mods,
        );
    }

    fn scroll(&self, cx: Scope, down: bool, count: usize, mods: Modifiers) {
        let config = self.common.config.get_untracked();
        let viewport = self.viewport.get_untracked();
        let line_height = config.editor.line_height() as f64;
        let diff = line_height * count as f64;
        let diff = if down { diff } else { -diff };

        let offset = self.cursor.with_untracked(|cursor| cursor.offset());
        let (line, _col) = self
            .doc
            .with_untracked(|doc| doc.buffer().offset_to_line_col(offset));
        let top = viewport.y0 + diff;
        let bottom = top + viewport.height();

        let new_line = if (line + 1) as f64 * line_height + line_height > bottom {
            let line = (bottom / line_height).floor() as usize;
            if line > 2 {
                line - 2
            } else {
                0
            }
        } else if line as f64 * line_height - line_height < top {
            let line = (top / line_height).ceil() as usize;
            line + 1
        } else {
            line
        };

        self.scroll_delta.set(Vec2::new(0.0, diff));

        match new_line.cmp(&line) {
            Ordering::Greater => {
                self.run_move_command(
                    cx,
                    &lapce_core::movement::Movement::Down,
                    Some(new_line - line),
                    mods,
                );
            }
            Ordering::Less => {
                self.run_move_command(
                    cx,
                    &lapce_core::movement::Movement::Up,
                    Some(line - new_line),
                    mods,
                );
            }
            _ => (),
        };
    }

    fn select_completion(&self, cx: Scope) {
        let item = self
            .common
            .completion
            .with_untracked(|c| c.current_item().cloned());
        self.cancel_completion();
        if let Some(item) = item {
            if item.item.data.is_some() {
                let editor = self.clone();
                let (rev, path) = self
                    .doc
                    .with_untracked(|doc| (doc.rev(), doc.content.path().cloned()));
                let offset = self.cursor.with_untracked(|c| c.offset());
                let send = create_ext_action(cx, move |item| {
                    if editor.cursor.with_untracked(|c| c.offset() != offset) {
                        return;
                    }
                    if editor.doc.with_untracked(|doc| {
                        doc.rev() != rev || doc.content.path() != path.as_ref()
                    }) {
                        return;
                    }
                    let _ = editor.apply_completion_item(&item);
                });
                self.common.proxy.completion_resolve(
                    item.plugin_id,
                    item.item.clone(),
                    move |result| {
                        let item =
                            if let Ok(ProxyResponse::CompletionResolveResponse {
                                item,
                            }) = result
                            {
                                *item
                            } else {
                                item.item.clone()
                            };
                        send(item);
                    },
                );
            } else {
                let _ = self.apply_completion_item(&item.item);
            }
        }
    }

    pub fn cancel_completion(&self) {
        if self.common.completion.with_untracked(|c| c.status)
            == CompletionStatus::Inactive
        {
            return;
        }
        self.common.completion.update(|c| {
            c.cancel();
        });
    }

    /// Update the displayed autocompletion box
    /// Sends a request to the LSP for completion information
    fn update_completion(&self, display_if_empty_input: bool) {
        if self.get_mode() != Mode::Insert {
            self.cancel_completion();
            return;
        }

        let path = match self.doc.with_untracked(|doc| {
            if doc.loaded() {
                doc.content.path().cloned()
            } else {
                None
            }
        }) {
            Some(path) => path,
            None => return,
        };

        let offset = self.cursor.with_untracked(|c| c.offset());
        let (start_offset, input, char) = self.doc.with_untracked(|doc| {
            let start_offset = doc.buffer().prev_code_boundary(offset);
            let end_offset = doc.buffer().next_code_boundary(offset);
            let input = doc
                .buffer()
                .slice_to_cow(start_offset..end_offset)
                .to_string();
            let char = if start_offset == 0 {
                "".to_string()
            } else {
                doc.buffer()
                    .slice_to_cow(start_offset - 1..start_offset)
                    .to_string()
            };
            (start_offset, input, char)
        });
        if !display_if_empty_input && input.is_empty() && char != "." && char != ":"
        {
            self.common.completion.update(|c| {
                c.cancel();
            });
            return;
        }

        if self.common.completion.with_untracked(|completion| {
            completion.status != CompletionStatus::Inactive
                && completion.offset == start_offset
                && completion.path == path
        }) {
            self.common.completion.update(|completion| {
                completion.update_input(input.clone());

                if !completion.input_items.contains_key("") {
                    let start_pos = self.doc.with_untracked(|doc| {
                        doc.buffer().offset_to_position(start_offset)
                    });
                    completion.request(
                        &self.common.proxy,
                        path.clone(),
                        "".to_string(),
                        start_pos,
                    );
                }

                if !completion.input_items.contains_key(&input) {
                    let position = self.doc.with_untracked(|doc| {
                        doc.buffer().offset_to_position(offset)
                    });
                    completion.request(&self.common.proxy, path, input, position);
                }
            });
            return;
        }

        self.common.completion.update(|completion| {
            completion.path = path.clone();
            completion.offset = start_offset;
            completion.input = input.clone();
            completion.status = CompletionStatus::Started;
            completion.input_items.clear();
            completion.request_id += 1;
            let start_pos = self
                .doc
                .with_untracked(|doc| doc.buffer().offset_to_position(start_offset));
            completion.request(
                &self.common.proxy,
                path.clone(),
                "".to_string(),
                start_pos,
            );

            if !input.is_empty() {
                let position = self
                    .doc
                    .with_untracked(|doc| doc.buffer().offset_to_position(offset));
                completion.request(&self.common.proxy, path, input, position);
            }
        });
    }

    /// Check if there are completions that are being rendered
    fn has_completions(&self) -> bool {
        self.common.completion.with_untracked(|completion| {
            completion.status != CompletionStatus::Inactive
                && !completion.filtered_items.is_empty()
        })
    }

    fn apply_completion_item(&self, item: &CompletionItem) -> Result<()> {
        let doc = self.doc.get_untracked();
        let cursor = self.cursor.get_untracked();
        // Get all the edits which would be applied in places other than right where the cursor is
        let additional_edit: Vec<_> = item
            .additional_text_edits
            .as_ref()
            .into_iter()
            .flatten()
            .map(|edit| {
                let selection = lapce_core::selection::Selection::region(
                    doc.buffer().offset_of_position(&edit.range.start),
                    doc.buffer().offset_of_position(&edit.range.end),
                );
                (selection, edit.new_text.as_str())
            })
            .collect::<Vec<(lapce_core::selection::Selection, &str)>>();

        let text_format = item
            .insert_text_format
            .unwrap_or(lsp_types::InsertTextFormat::PLAIN_TEXT);
        if let Some(edit) = &item.text_edit {
            match edit {
                CompletionTextEdit::Edit(edit) => {
                    let offset = cursor.offset();
                    let start_offset = doc.buffer().prev_code_boundary(offset);
                    let end_offset = doc.buffer().next_code_boundary(offset);
                    let edit_start =
                        doc.buffer().offset_of_position(&edit.range.start);
                    let edit_end = doc.buffer().offset_of_position(&edit.range.end);

                    let selection = lapce_core::selection::Selection::region(
                        start_offset.min(edit_start),
                        end_offset.max(edit_end),
                    );
                    match text_format {
                        lsp_types::InsertTextFormat::PLAIN_TEXT => {
                            self.do_edit(
                                &selection,
                                &[
                                    &[(selection.clone(), edit.new_text.as_str())][..],
                                    &additional_edit[..],
                                ]
                                .concat(),
                            );
                            return Ok(());
                        }
                        lsp_types::InsertTextFormat::SNIPPET => {
                            self.completion_apply_snippet(
                                &edit.new_text,
                                &selection,
                                additional_edit,
                                start_offset,
                            )?;
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                CompletionTextEdit::InsertAndReplace(_) => (),
            }
        }

        let offset = cursor.offset();
        let start_offset = doc.buffer().prev_code_boundary(offset);
        let end_offset = doc.buffer().next_code_boundary(offset);
        let selection = Selection::region(start_offset, end_offset);

        self.do_edit(
            &selection,
            &[
                &[(
                    selection.clone(),
                    item.insert_text.as_deref().unwrap_or(item.label.as_str()),
                )][..],
                &additional_edit[..],
            ]
            .concat(),
        );
        Ok(())
    }

    fn completion_apply_snippet(
        &self,
        snippet: &str,
        selection: &Selection,
        additional_edit: Vec<(Selection, &str)>,
        start_offset: usize,
    ) -> Result<()> {
        let snippet = Snippet::from_str(snippet)?;
        let text = snippet.text();
        let mut cursor = self.cursor.get_untracked();
        let old_cursor = cursor.mode.clone();
        let (delta, inval_lines, edits) = self
            .doc
            .try_update(|doc| {
                doc.do_raw_edit(
                    &[
                        &[(selection.clone(), text.as_str())][..],
                        &additional_edit[..],
                    ]
                    .concat(),
                    EditType::Completion,
                )
            })
            .unwrap();

        let selection = selection.apply_delta(&delta, true, InsertDrift::Default);

        let mut transformer = Transformer::new(&delta);
        let offset = transformer.transform(start_offset, false);
        let snippet_tabs = snippet.tabs(offset);

        if snippet_tabs.is_empty() {
            self.doc.update(|doc| {
                cursor.update_selection(doc.buffer(), selection);
                doc.buffer_mut().set_cursor_before(old_cursor);
                doc.buffer_mut().set_cursor_after(cursor.mode.clone());
            });
            self.cursor.set(cursor);
            self.apply_deltas(&[(delta, inval_lines, edits)]);
            return Ok(());
        }

        let mut selection = lapce_core::selection::Selection::new();
        let (_tab, (start, end)) = &snippet_tabs[0];
        let region = lapce_core::selection::SelRegion::new(*start, *end, None);
        selection.add_region(region);
        cursor.set_insert(selection);

        self.doc.update(|doc| {
            doc.buffer_mut().set_cursor_before(old_cursor);
            doc.buffer_mut().set_cursor_after(cursor.mode.clone());
        });
        self.cursor.set(cursor);
        self.apply_deltas(&[(delta, inval_lines, edits)]);
        self.add_snippet_placeholders(snippet_tabs);
        Ok(())
    }

    fn add_snippet_placeholders(
        &self,
        new_placeholders: Vec<(usize, (usize, usize))>,
    ) {
        self.snippet.update(|snippet| {
            if snippet.is_none() {
                if new_placeholders.len() > 1 {
                    *snippet = Some(new_placeholders);
                }
                return;
            }

            let placeholders = snippet.as_mut().unwrap();

            let mut current = 0;
            let offset = self.cursor.get_untracked().offset();
            for (i, (_, (start, end))) in placeholders.iter().enumerate() {
                if *start <= offset && offset <= *end {
                    current = i;
                    break;
                }
            }

            let v = placeholders.split_off(current);
            placeholders.extend_from_slice(&new_placeholders);
            placeholders.extend_from_slice(&v[1..]);
        });
    }

    fn do_edit(
        &self,
        selection: &Selection,
        edits: &[(impl AsRef<Selection>, &str)],
    ) {
        let mut cursor = self.cursor.get_untracked();
        let (delta, inval_lines, edits) = self
            .doc
            .try_update(|doc| {
                let (delta, inval_lines, edits) =
                    doc.do_raw_edit(edits, EditType::Completion);
                let selection =
                    selection.apply_delta(&delta, true, InsertDrift::Default);
                let old_cursor = cursor.mode.clone();
                cursor.update_selection(doc.buffer(), selection);
                doc.buffer_mut().set_cursor_before(old_cursor);
                doc.buffer_mut().set_cursor_after(cursor.mode.clone());
                (delta, inval_lines, edits)
            })
            .unwrap();
        self.cursor.set(cursor);

        self.apply_deltas(&[(delta, inval_lines, edits)]);
    }

    pub fn do_text_edit(&self, edits: &[TextEdit]) {
        let (selection, edits) = self.doc.with_untracked(|doc| {
            let selection = self.cursor.get_untracked().edit_selection(doc.buffer());
            let edits = edits
                .iter()
                .map(|edit| {
                    let selection = lapce_core::selection::Selection::region(
                        doc.buffer().offset_of_position(&edit.range.start),
                        doc.buffer().offset_of_position(&edit.range.end),
                    );
                    (selection, edit.new_text.as_str())
                })
                .collect::<Vec<_>>();
            (selection, edits)
        });

        self.do_edit(&selection, &edits);
    }

    fn apply_deltas(&self, deltas: &[(RopeDelta, InvalLines, SyntaxEdit)]) {
        if !deltas.is_empty() && !self.confirmed.get_untracked() {
            self.confirmed.set(true);
        }
        for (delta, _, _) in deltas {
            // self.inactive_apply_delta(delta);
            self.update_snippet_offset(delta);
            // self.update_breakpoints(delta);
        }
        // self.update_signature();
    }

    fn update_snippet_offset(&self, delta: &RopeDelta) {
        if self.snippet.with_untracked(|s| s.is_some()) {
            self.snippet.update(|snippet| {
                let mut transformer = Transformer::new(delta);
                *snippet = Some(
                    snippet
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|(tab, (start, end))| {
                            (
                                *tab,
                                (
                                    transformer.transform(*start, false),
                                    transformer.transform(*end, true),
                                ),
                            )
                        })
                        .collect(),
                );
            });
        }
    }

    fn do_go_to_location(
        &self,
        cx: Scope,
        location: EditorLocation,
        edits: Option<Vec<TextEdit>>,
    ) {
        if let Some(position) = location.position {
            self.go_to_position(position, location.scroll_offset, edits);
        } else if let Some(edits) = edits.as_ref() {
            self.do_text_edit(edits);
        } else {
            let db: Arc<LapceDb> = use_context(cx).unwrap();
            if let Ok(info) = db.get_doc_info(&self.common.workspace, &location.path)
            {
                self.go_to_position(
                    EditorPosition::Offset(info.cursor_offset),
                    Some(Vec2::new(info.scroll_offset.0, info.scroll_offset.1)),
                    edits,
                );
            }
        }
    }

    pub fn go_to_location(
        &self,
        cx: Scope,
        location: EditorLocation,
        new_doc: bool,
        edits: Option<Vec<TextEdit>>,
    ) {
        if !new_doc {
            self.do_go_to_location(cx, location, edits);
        } else {
            let buffer_id = self.doc.with_untracked(|doc| doc.buffer_id);
            let set_doc = self.doc.write_only();
            let editor = self.clone();
            let path = location.path.clone();
            let send = create_ext_action(cx, move |content| {
                set_doc.update(move |doc| {
                    doc.init_content(content);
                });

                editor.do_go_to_location(cx, location.clone(), edits.clone());
            });

            self.common
                .proxy
                .new_buffer(buffer_id, path, move |result| {
                    if let Ok(ProxyResponse::NewBufferResponse { content }) = result
                    {
                        send(Rope::from(content))
                    }
                });
        }
    }

    pub fn go_to_position(
        &self,
        position: EditorPosition,
        scroll_offset: Option<Vec2>,
        edits: Option<Vec<TextEdit>>,
    ) {
        let offset = self
            .doc
            .with_untracked(|doc| position.to_offset(doc.buffer()));
        let config = self.common.config.get_untracked();
        // TODO: are these the right affinities?
        self.cursor.set(if config.core.modal {
            Cursor::new(
                CursorMode::Normal(offset),
                None,
                None,
                CursorAffinity::Forward,
            )
        } else {
            Cursor::new(
                CursorMode::Insert(Selection::caret(offset)),
                None,
                None,
                CursorAffinity::Forward,
            )
        });
        if let Some(scroll_offset) = scroll_offset {
            self.scroll_to.set(Some(scroll_offset));
        }
        if let Some(edits) = edits.as_ref() {
            self.do_text_edit(edits);
        }
    }

    pub fn get_code_actions(&self, cx: Scope) {
        let path = match self.doc.with_untracked(|doc| {
            if doc.loaded() {
                doc.content.path().cloned()
            } else {
                None
            }
        }) {
            Some(path) => path,
            None => return,
        };

        let offset = self.cursor.with_untracked(|c| c.offset());
        let exists = self
            .doc
            .with_untracked(|doc| doc.code_actions.contains_key(&offset));

        if exists {
            return;
        }

        self.doc.update(|doc| {
            // insert some empty data, so that we won't make the request again
            doc.code_actions
                .insert(offset, Arc::new((PluginId(0), Vec::new())));
        });

        let (position, rev, diagnostics) = self.doc.with_untracked(|doc| {
            let position = doc.buffer().offset_to_position(offset);
            let rev = doc.rev();

            // Get the diagnostics for the current line, which the LSP might use to inform
            // what code actions are available (such as fixes for the diagnostics).
            let diagnostics = doc
                .diagnostics
                .diagnostics
                .get_untracked()
                .iter()
                .map(|x| &x.diagnostic)
                .filter(|x| {
                    x.range.start.line <= position.line
                        && x.range.end.line >= position.line
                })
                .cloned()
                .collect();

            (position, rev, diagnostics)
        });

        let doc = self.doc;
        let send = create_ext_action(cx, move |resp| {
            if doc.with_untracked(|doc| doc.rev() == rev) {
                doc.update(|doc| {
                    doc.code_actions.insert(offset, Arc::new(resp));
                });
            }
        });

        self.common.proxy.get_code_actions(
            path,
            position,
            diagnostics,
            move |result| {
                if let Ok(ProxyResponse::GetCodeActionsResponse {
                    plugin_id,
                    resp,
                }) = result
                {
                    send((plugin_id, resp))
                }
            },
        );
    }

    pub fn show_code_actions(&self, mouse_click: bool) {
        let offset = self.cursor.with_untracked(|c| c.offset());
        let code_actions = self
            .doc
            .with_untracked(|doc| doc.code_actions.get(&offset).cloned());
        if let Some(code_actions) = code_actions {
            if !code_actions.1.is_empty() {
                self.common.internal_command.set(Some(
                    InternalCommand::ShowCodeActions {
                        offset,
                        mouse_click,
                        code_actions,
                    },
                ));
            }
        }
    }

    fn do_save(&self, cx: Scope) {
        let (rev, content) = self
            .doc
            .with_untracked(|doc| (doc.rev(), doc.content.clone()));

        let doc = self.doc;
        let send = create_ext_action(cx, move |result| {
            if let Ok(ProxyResponse::SaveResponse {}) = result {
                let current_rev = doc.with_untracked(|doc| doc.rev());
                if current_rev == rev {
                    doc.update(|doc| {
                        doc.buffer_mut().set_pristine();
                    });
                }
            }
        });

        if let DocContent::File(path) = content {
            self.common.proxy.save(rev, path, move |result| {
                send(result);
            })
        }
    }

    fn save(&self, cx: Scope, exit: bool, allow_formatting: bool) {
        let (rev, is_pristine, content) = self.doc.with_untracked(|doc| {
            (doc.rev(), doc.buffer().is_pristine(), doc.content.clone())
        });

        if content.path().is_some() && is_pristine {
            if exit {}
            return;
        }

        let config = self.common.config.get_untracked();
        if let DocContent::File(path) = content {
            let format_on_save = allow_formatting && config.editor.format_on_save;
            if format_on_save {
                let editor = self.clone();
                let send = create_ext_action(cx, move |result| {
                    if let Ok(Ok(ProxyResponse::GetDocumentFormatting { edits })) =
                        result
                    {
                        let current_rev = editor.doc.with_untracked(|doc| doc.rev());
                        if current_rev == rev {
                            editor.do_text_edit(&edits);
                        }
                    }
                    editor.do_save(cx);
                });

                let (tx, rx) = crossbeam_channel::bounded(1);
                let proxy = self.common.proxy.clone();
                std::thread::spawn(move || {
                    proxy.get_document_formatting(path, move |result| {
                        let _ = tx.send(result);
                    });
                    let result = rx.recv_timeout(std::time::Duration::from_secs(1));
                    send(result);
                });
            } else {
                self.do_save(cx);
            }
        }
    }

    fn search_whole_word_forward(&self, cx: Scope, mods: Modifiers) {
        let offset = self.cursor.with_untracked(|c| c.offset());
        let (word, buffer) = self.doc.with_untracked(|doc| {
            let (start, end) = doc.buffer().select_word(offset);
            (
                doc.buffer().slice_to_cow(start..end).to_string(),
                doc.buffer().clone(),
            )
        });
        let next = self
            .find
            .try_update(|find| {
                find.visual = true;
                find.set_find(&word, false, true);
                find.next(buffer.text(), offset, false, true)
            })
            .unwrap();

        if let Some((start, _end)) = next {
            self.run_move_command(
                cx,
                &lapce_core::movement::Movement::Offset(start),
                None,
                mods,
            );
        }
    }

    fn search_forward(&self, cx: Scope, mods: Modifiers) {
        let offset = self.cursor.with_untracked(|c| c.offset());
        let buffer = self.doc.with_untracked(|doc| doc.buffer().clone());
        let next = self
            .find
            .try_update(|find| {
                find.visual = true;
                find.next(buffer.text(), offset, false, true)
            })
            .unwrap();

        if let Some((start, _end)) = next {
            self.run_move_command(
                cx,
                &lapce_core::movement::Movement::Offset(start),
                None,
                mods,
            );
        }
    }

    fn search_backward(&self, cx: Scope, mods: Modifiers) {
        let offset = self.cursor.with_untracked(|c| c.offset());
        let buffer = self.doc.with_untracked(|doc| doc.buffer().clone());
        let next = self
            .find
            .try_update(|find| {
                find.visual = true;
                find.next(buffer.text(), offset, true, true)
            })
            .unwrap();

        if let Some((start, _end)) = next {
            self.run_move_command(
                cx,
                &lapce_core::movement::Movement::Offset(start),
                None,
                mods,
            );
        }
    }

    pub fn save_doc_position(&self, cx: Scope) {
        let path = match self.doc.with_untracked(|doc| {
            if doc.loaded() {
                doc.content.path().cloned()
            } else {
                None
            }
        }) {
            Some(path) => path,
            None => return,
        };

        let cursor_offset = self.cursor.with_untracked(|c| c.offset());
        let scroll_offset = self.viewport.with_untracked(|v| v.origin().to_vec2());

        let db: Arc<LapceDb> = use_context(cx).unwrap();
        db.save_doc_position(
            &self.common.workspace,
            path,
            cursor_offset,
            scroll_offset,
        );
    }
}

impl KeyPressFocus for EditorData {
    fn get_mode(&self) -> lapce_core::mode::Mode {
        self.cursor.with_untracked(|c| c.get_mode())
    }

    fn check_condition(
        &self,
        condition: crate::keypress::condition::Condition,
    ) -> bool {
        match condition {
            Condition::ListFocus => self.has_completions(),
            Condition::CompletionFocus => self.has_completions(),
            Condition::InSnippet => self.snippet.with_untracked(|s| s.is_some()),
            Condition::EditorFocus => {
                self.doc.with_untracked(|doc| !doc.content.is_local())
            }
            _ => false,
        }
    }

    fn run_command(
        &self,
        cx: Scope,
        command: &crate::command::LapceCommand,
        count: Option<usize>,
        mods: floem::glazier::Modifiers,
    ) -> crate::command::CommandExecuted {
        match &command.kind {
            crate::command::CommandKind::Workbench(_) => CommandExecuted::No,
            crate::command::CommandKind::Edit(cmd) => self.run_edit_command(cx, cmd),
            crate::command::CommandKind::Move(cmd) => {
                let movement = cmd.to_movement(count);
                self.run_move_command(cx, &movement, count, mods)
            }
            crate::command::CommandKind::Focus(cmd) => {
                self.run_focus_command(cx, cmd, count, mods)
            }
            crate::command::CommandKind::MotionMode(_) => CommandExecuted::No,
            crate::command::CommandKind::MultiSelection(_) => CommandExecuted::No,
        }
    }

    fn expect_char(&self) -> bool {
        self.inline_find.with_untracked(|f| f.is_some())
    }

    fn receive_char(&self, cx: Scope, c: &str) {
        if self.get_mode() == Mode::Insert {
            let mut cursor = self.cursor.get_untracked();
            let config = self.common.config.get_untracked();
            let deltas = self
                .doc
                .try_update(|doc| doc.do_insert(&mut cursor, c, &config))
                .unwrap();
            self.cursor.set(cursor);

            if !c
                .chars()
                .all(|c| c.is_whitespace() || c.is_ascii_whitespace())
            {
                self.update_completion(false);
            } else {
                self.cancel_completion();
            }
            self.apply_deltas(&deltas);
        } else if let Some(direction) = self.inline_find.get_untracked() {
            self.inline_find(cx, direction.clone(), c);
            self.last_inline_find.set(Some((direction, c.to_string())));
            self.inline_find.set(None);
        }
    }
}

/// Checks if completion should be triggered if the received command
/// is one that inserts whitespace or deletes whitespace
fn show_completion(
    cmd: &EditCommand,
    doc: &Rope,
    deltas: &[(RopeDelta, InvalLines, SyntaxEdit)],
) -> bool {
    let show_completion = match cmd {
        EditCommand::DeleteBackward
        | EditCommand::DeleteForward
        | EditCommand::DeleteWordBackward
        | EditCommand::DeleteWordForward
        | EditCommand::DeleteForwardAndInsert => {
            let start = match deltas.get(0).and_then(|delta| delta.0.els.get(0)) {
                Some(lapce_xi_rope::DeltaElement::Copy(_, start)) => *start,
                _ => 0,
            };

            let end = match deltas.get(0).and_then(|delta| delta.0.els.get(1)) {
                Some(lapce_xi_rope::DeltaElement::Copy(end, _)) => *end,
                _ => 0,
            };

            if start > 0 && end > start {
                !doc.slice_to_cow(start..end)
                    .chars()
                    .all(|c| c.is_whitespace() || c.is_ascii_whitespace())
            } else {
                true
            }
        }
        _ => false,
    };

    show_completion
}
