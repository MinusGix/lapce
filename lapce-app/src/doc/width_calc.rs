use floem::cosmic_text::{Attrs, AttrsList, FamilyOwned, TextLayout};
use smallvec::SmallVec;

pub trait WidthCalc {
    fn measure_width(&self, text: &str) -> f64;
}

#[derive(Debug, Clone)]
pub struct BasicWidthCalc {
    family: SmallVec<[FamilyOwned; 3]>,
    font_size: f32,
}
impl BasicWidthCalc {
    pub fn new(family: SmallVec<[FamilyOwned; 3]>, font_size: f32) -> Self {
        Self { family, font_size }
    }
}
impl WidthCalc for BasicWidthCalc {
    fn measure_width(&self, text: &str) -> f64 {
        let attrs = Attrs::new().family(&self.family).font_size(self.font_size);
        let attrs_list = AttrsList::new(attrs);
        let mut layout = TextLayout::new();
        layout.set_text(text, attrs_list);

        layout.size().width
    }
}

#[derive(Debug, Clone, Default)]
pub struct ByteWidthCalc;
impl WidthCalc for ByteWidthCalc {
    fn measure_width(&self, text: &str) -> f64 {
        text.len() as f64
    }
}

// TODO: once we make wrapping pay attention to phantom text, we'll need to include that in the info, since it can have different font sizes and font families

// TODO: We might want a width calc that caches the values
