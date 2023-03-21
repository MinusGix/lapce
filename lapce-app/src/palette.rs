use std::{
    cell::RefCell,
    collections::HashMap,
    path::PathBuf,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use floem::{
    app::AppContext,
    ext_event::{create_ext_action, create_signal_from_channel},
    reactive::{
        create_effect, create_rw_signal, create_signal, ReadSignal, RwSignal,
        SignalGet, SignalGetUntracked, SignalSet, SignalUpdate, SignalWith,
        SignalWithUntracked, WriteSignal,
    },
};
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use itertools::Itertools;
use lapce_core::{
    command::FocusCommand, mode::Mode, movement::Movement, register::Register,
    selection::Selection,
};
use lapce_rpc::proxy::{ProxyResponse, ProxyRpcHandler};
use lapce_xi_rope::Rope;

use crate::{
    command::{CommandExecuted, CommandKind, InternalCommand},
    completion::CompletionData,
    config::LapceConfig,
    editor::EditorData,
    id::EditorId,
    keypress::{condition::Condition, KeyPressData, KeyPressFocus},
    window_tab::Focus,
    workspace::LapceWorkspace,
};

use self::{
    item::{PaletteItem, PaletteItemContent},
    kind::PaletteKind,
};

pub mod item;
pub mod kind;

#[derive(Clone, PartialEq, Eq)]
pub enum PaletteStatus {
    Inactive,
    Started,
    Done,
}

#[derive(Clone, Debug)]
pub struct PaletteInput {
    pub input: String,
    pub kind: PaletteKind,
}

impl PaletteInput {
    pub fn update_input(&mut self, input: String) {
        self.kind = self.kind.get_palette_kind(&input);
        self.input = self.kind.get_input(&input).to_string();
    }
}

#[derive(Clone)]
pub struct PaletteData {
    run_id_counter: Arc<AtomicU64>,
    run_tx: Sender<(u64, String, im::Vector<PaletteItem>)>,
    internal_command: WriteSignal<Option<InternalCommand>>,
    pub run_id: RwSignal<u64>,
    pub workspace: Arc<LapceWorkspace>,
    pub status: RwSignal<PaletteStatus>,
    pub index: RwSignal<usize>,
    pub items: RwSignal<im::Vector<PaletteItem>>,
    pub filtered_items: ReadSignal<im::Vector<PaletteItem>>,
    pub proxy_rpc: ProxyRpcHandler,
    pub input: RwSignal<PaletteInput>,
    pub editor: EditorData,
    pub focus: RwSignal<Focus>,
    pub keypress: ReadSignal<KeyPressData>,
    pub config: ReadSignal<Arc<LapceConfig>>,
    pub executed_commands: Rc<RefCell<HashMap<String, Instant>>>,
}

impl PaletteData {
    pub fn new(
        cx: AppContext,
        workspace: Arc<LapceWorkspace>,
        proxy_rpc: ProxyRpcHandler,
        register: RwSignal<Register>,
        completion: RwSignal<CompletionData>,
        internal_command: WriteSignal<Option<InternalCommand>>,
        focus: RwSignal<Focus>,
        keypress: ReadSignal<KeyPressData>,
        config: ReadSignal<Arc<LapceConfig>>,
    ) -> Self {
        let status = create_rw_signal(cx.scope, PaletteStatus::Inactive);
        let items = create_rw_signal(cx.scope, im::Vector::new());
        let index = create_rw_signal(cx.scope, 0);
        let input = create_rw_signal(
            cx.scope,
            PaletteInput {
                input: "".to_string(),
                kind: PaletteKind::File,
            },
        );
        let editor = EditorData::new_local(
            cx,
            EditorId::next(),
            register,
            completion,
            internal_command,
            proxy_rpc.clone(),
            config,
        );
        let run_id = create_rw_signal(cx.scope, 0);
        let run_id_counter = Arc::new(AtomicU64::new(0));

        let (run_tx, run_rx) = crossbeam_channel::unbounded();
        {
            let run_id = run_id.read_only();
            let input = input.read_only();
            let items = items.read_only();
            let tx = run_tx.clone();

            {
                let tx = tx.clone();
                // this effect only monitors items change
                create_effect(cx.scope, move |_| {
                    let items = items.get();
                    println!("filter items when items change");
                    let input = input.get_untracked();
                    let run_id = run_id.get_untracked();
                    let _ = tx.send((run_id, input.input, items));
                });
            }

            // this effect only monitors input change
            create_effect(cx.scope, move |last_kind| {
                let input = input.get();
                println!("new input {input:?}");
                let kind = input.kind;
                if last_kind != Some(kind) {
                    println!("got new kind {kind:?}");
                    return kind;
                }

                println!("filter items when input change");
                let items = items.get_untracked();
                let run_id = run_id.get_untracked();
                let _ = tx.send((run_id, input.input, items));
                kind
            });
        }

        let (resp_tx, resp_rx) = crossbeam_channel::unbounded();
        {
            let run_id = run_id_counter.clone();
            std::thread::spawn(move || {
                Self::update_process(run_id, run_rx, resp_tx);
            });
        }

        let (filtered_items, set_filtered_items) =
            create_signal(cx.scope, im::Vector::new());
        {
            let resp = create_signal_from_channel(cx, resp_rx);
            let run_id = run_id.read_only();
            let input = input.read_only();
            let index = index.write_only();
            create_effect(cx.scope, move |_| {
                if let Some((filter_run_id, filter_input, items)) = resp.get() {
                    if run_id.get_untracked() == filter_run_id
                        && input.get_untracked().input == filter_input
                    {
                        set_filtered_items.set(items);
                        index.set(0);
                    }
                }
            });
        }

        let palette = Self {
            run_id_counter,
            run_tx,
            internal_command,
            run_id,
            focus,
            workspace,
            status,
            index,
            items,
            filtered_items,
            editor,
            input,
            proxy_rpc,
            keypress,
            config,
            executed_commands: Rc::new(RefCell::new(HashMap::new())),
        };

        {
            let palette = palette.clone();
            let doc = palette.editor.doc.read_only();
            let input = palette.input.write_only();
            let status = palette.status.read_only();
            // this effect monitors the document change in the palette input editor
            create_effect(cx.scope, move |last_input| {
                let new_input = doc.with(|doc| doc.buffer().text().to_string());
                let status = status.get_untracked();
                if status == PaletteStatus::Inactive {
                    // If the status is inactive, we set the input to None,
                    // so that when we actually run the palette, the input
                    // can be compared with this None.
                    return None;
                }

                let last_input_is_none = !matches!(last_input, Some(Some(_)));

                let changed = match last_input {
                    None => true,
                    Some(last_input) => {
                        Some(new_input.as_str()) != last_input.as_deref()
                    }
                };

                if changed {
                    let new_kind = input
                        .try_update(|input| {
                            let kind = input.kind;
                            input.update_input(new_input.clone());
                            if last_input_is_none || kind != input.kind {
                                Some(input.kind)
                            } else {
                                None
                            }
                        })
                        .unwrap();
                    if let Some(new_kind) = new_kind {
                        palette.run_inner(cx, new_kind);
                    }
                }
                Some(new_input)
            });
        }

        palette
    }

    pub fn run(&self, cx: AppContext, kind: PaletteKind) {
        self.status.set(PaletteStatus::Started);
        let symbol = kind.symbol();
        self.editor
            .doc
            .update(|doc| doc.reload(Rope::from(symbol), true));
        self.editor
            .cursor
            .update(|cursor| cursor.set_insert(Selection::caret(symbol.len())));
    }

    fn run_inner(&self, cx: AppContext, kind: PaletteKind) {
        println!("palette run inner");
        let run_id = self.run_id_counter.fetch_add(1, Ordering::Relaxed) + 1;
        self.run_id.set(run_id);
        match kind {
            PaletteKind::File => {
                self.get_files(cx);
            }
            PaletteKind::Command => {
                self.get_commands(cx);
            }
        }
    }

    fn get_files(&self, cx: AppContext) {
        let workspace = self.workspace.clone();
        let set_items = self.items.write_only();
        let send = create_ext_action(cx, move |items: Vec<PathBuf>| {
            let items = items
                .into_iter()
                .map(|path| {
                    let full_path = path.clone();
                    let mut path = path;
                    if let Some(workspace_path) = workspace.path.as_ref() {
                        path = path
                            .strip_prefix(workspace_path)
                            .unwrap_or(&full_path)
                            .to_path_buf();
                    }
                    let filter_text = path.to_str().unwrap_or("").to_string();
                    PaletteItem {
                        content: PaletteItemContent::File { path, full_path },
                        filter_text,
                        score: 0,
                        indices: Vec::new(),
                    }
                })
                .collect::<im::Vector<_>>();
            set_items.set(items);
        });
        self.proxy_rpc.get_files(move |result| {
            if let Ok(ProxyResponse::GetFilesResponse { items }) = result {
                send(items);
            }
        });
    }

    fn get_commands(&self, cx: AppContext) {
        const EXCLUDED_ITEMS: &[&str] = &["palette.command"];

        self.keypress.get_untracked();
        let items = self.keypress.with_untracked(|keypress| {
            let mut i = 0;
            let mut items: im::Vector<PaletteItem> = self
                .executed_commands
                .borrow()
                .iter()
                .sorted_by_key(|(_, i)| *i)
                .rev()
                .filter_map(|(key, _)| {
                    keypress.commands.get(key).and_then(|c| {
                        c.kind.desc().as_ref().map(|m| {
                            let item = PaletteItem {
                                content: PaletteItemContent::Command {
                                    cmd: c.clone(),
                                },
                                filter_text: m.to_string(),
                                score: 0,
                                indices: vec![],
                            };
                            i += 1;
                            item
                        })
                    })
                })
                .collect();
            items.extend(keypress.commands.iter().filter_map(|(_, c)| {
                if EXCLUDED_ITEMS.contains(&c.kind.str()) {
                    return None;
                }

                if self.executed_commands.borrow().contains_key(c.kind.str()) {
                    return None;
                }

                c.kind.desc().as_ref().map(|m| {
                    let item = PaletteItem {
                        content: PaletteItemContent::Command { cmd: c.clone() },
                        filter_text: m.to_string(),
                        score: 0,
                        indices: vec![],
                    };
                    i += 1;
                    item
                })
            }));

            items
        });

        self.items.set(items);
    }

    fn select(&self, cx: AppContext) {
        let index = self.index.get_untracked();
        let items = self.filtered_items.get_untracked();
        if let Some(item) = items.get(index) {
            match &item.content {
                PaletteItemContent::File { full_path, .. } => {
                    self.internal_command.set(Some(InternalCommand::OpenFile {
                        path: full_path.to_owned(),
                    }));
                }
                PaletteItemContent::Command { cmd } => {}
            }
        }
        self.cancel(cx);
    }

    fn cancel(&self, cx: AppContext) {
        self.status.set(PaletteStatus::Inactive);
        self.focus.set(Focus::Workbench);
        self.items.update(|items| items.clear());
        self.editor
            .doc
            .update(|doc| doc.reload(Rope::from(""), true));
        self.editor
            .cursor
            .update(|cursor| cursor.set_insert(Selection::caret(0)));
    }

    fn next(&self) {
        let index = self.index.get_untracked();
        let len = self.filtered_items.with(|i| i.len());
        let new_index = Movement::Down.update_index(index, len, 1, true);
        self.index.set(new_index);
    }

    fn previous(&self) {
        let index = self.index.get_untracked();
        let len = self.filtered_items.with(|i| i.len());
        let new_index = Movement::Up.update_index(index, len, 1, true);
        self.index.set(new_index);
    }

    fn next_page(&self) {}

    fn previous_page(&self) {}

    fn run_focus_command(
        &self,
        cx: AppContext,
        cmd: &FocusCommand,
    ) -> CommandExecuted {
        match cmd {
            FocusCommand::ModalClose => {
                self.cancel(cx);
            }
            FocusCommand::ListNext => {
                self.next();
            }
            FocusCommand::ListNextPage => {
                self.next_page();
            }
            FocusCommand::ListPrevious => {
                self.previous();
            }
            FocusCommand::ListPreviousPage => {
                self.previous_page();
            }
            FocusCommand::ListSelect => {
                self.select(cx);
            }
            _ => return CommandExecuted::No,
        }
        CommandExecuted::Yes
    }

    fn filter_items(
        run_id: Arc<AtomicU64>,
        current_run_id: u64,
        input: &str,
        items: im::Vector<PaletteItem>,
        matcher: &SkimMatcherV2,
    ) -> Option<im::Vector<PaletteItem>> {
        if input.is_empty() {
            return Some(items);
        }

        // Collecting into a Vec to sort we as are hitting a worst case in
        // `im::Vector` that leads to a stack overflow
        let mut filtered_items = Vec::new();
        for i in &items {
            if run_id.load(std::sync::atomic::Ordering::Acquire) != current_run_id {
                return None;
            }
            if let Some((score, indices)) =
                matcher.fuzzy_indices(&i.filter_text, input)
            {
                let mut item = i.clone();
                item.score = score;
                item.indices = indices;
                filtered_items.push(item);
            }
        }

        filtered_items.sort_by(|a, b| {
            let order = b.score.cmp(&a.score);
            match order {
                std::cmp::Ordering::Equal => a.filter_text.cmp(&b.filter_text),
                _ => order,
            }
        });

        if run_id.load(std::sync::atomic::Ordering::Acquire) != current_run_id {
            return None;
        }
        Some(filtered_items.into())
    }

    fn update_process(
        run_id: Arc<AtomicU64>,
        receiver: Receiver<(u64, String, im::Vector<PaletteItem>)>,
        resp_tx: Sender<(u64, String, im::Vector<PaletteItem>)>,
    ) {
        fn receive_batch(
            receiver: &Receiver<(u64, String, im::Vector<PaletteItem>)>,
        ) -> Result<(u64, String, im::Vector<PaletteItem>)> {
            let (mut run_id, mut input, mut items) = receiver.recv()?;
            loop {
                match receiver.try_recv() {
                    Ok(update) => {
                        run_id = update.0;
                        input = update.1;
                        items = update.2;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break,
                }
            }
            Ok((run_id, input, items))
        }

        let matcher = SkimMatcherV2::default().ignore_case();
        loop {
            if let Ok((current_run_id, input, items)) = receive_batch(&receiver) {
                if let Some(filtered_items) = Self::filter_items(
                    run_id.clone(),
                    current_run_id,
                    &input,
                    items,
                    &matcher,
                ) {
                    let _ = resp_tx.send((current_run_id, input, filtered_items));
                }
            } else {
                return;
            }
        }
    }
}

impl KeyPressFocus for PaletteData {
    fn get_mode(&self) -> lapce_core::mode::Mode {
        Mode::Insert
    }

    fn check_condition(
        &self,
        condition: crate::keypress::condition::Condition,
    ) -> bool {
        matches!(
            condition,
            Condition::ListFocus | Condition::PaletteFocus | Condition::ModalFocus
        )
    }

    fn run_command(
        &self,
        cx: AppContext,
        command: &crate::command::LapceCommand,
        count: Option<usize>,
        mods: floem::glazier::Modifiers,
    ) -> CommandExecuted {
        match &command.kind {
            CommandKind::Workbench(_) => todo!(),
            CommandKind::Edit(_) => {
                self.editor.run_command(cx, command, count, mods)
            }
            CommandKind::Move(_) => {
                self.editor.run_command(cx, command, count, mods)
            }
            CommandKind::Focus(cmd) => self.run_focus_command(cx, cmd),
            CommandKind::MotionMode(_) => todo!(),
            CommandKind::MultiSelection(_) => todo!(),
        }
    }

    fn receive_char(&self, cx: AppContext, c: &str) {
        self.editor.receive_char(cx, c);
    }
}
