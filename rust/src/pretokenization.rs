//! GPT-2 splitting over borrowed UTF-8 spans with fixed Unicode 17.0.0 tables.

use regex::Regex;

use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

#[derive(Clone, Copy)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub special: bool,
}

pub struct PreTokenizer {
    specials: Option<Regex>,
    special_width: usize,
}

impl PreTokenizer {
    pub fn new(mut specials: Vec<String>) -> Self {
        specials.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
        let special_width = specials.iter().map(String::len).max().unwrap_or(0);
        let pattern = specials
            .iter()
            .map(|s| regex::escape(s))
            .collect::<Vec<_>>()
            .join("|");
        Self {
            specials: (!specials.is_empty())
                .then(|| Regex::new(&pattern).expect("escaped literals")),
            special_width,
        }
    }

    fn class(&self, ch: char) -> usize {
        if ch.is_ascii() {
            return match ch {
                'a'..='z' | 'A'..='Z' => 0,
                '0'..='9' => 1,
                ' ' | '\t' | '\n' | '\r' | '\u{000b}' | '\u{000c}' => 2,
                _ => 3,
            };
        }
        match ch.general_category_group() {
            GeneralCategoryGroup::Letter => 0,
            GeneralCategoryGroup::Number => 1,
            GeneralCategoryGroup::Separator => 2,
            // Unicode White_Space adds TAB..CR and NEXT LINE to separators.
            // TAB..CR were handled above by the ASCII path.
            _ if ch == '\u{0085}' => 2,
            _ => 3,
        }
    }

    fn ordinary(&self, text: &str, start: usize, end: usize, out: &mut Vec<Span>) {
        let mut pos = start;
        while pos < end {
            let tail = &text[pos..end];
            if tail.starts_with('\'')
                && let Some(suffix) = ["'s", "'d", "'m", "'t", "'ll", "'ve", "'re"]
                    .iter()
                    .find(|suffix| tail.starts_with(**suffix))
            {
                out.push(Span {
                    start: pos,
                    end: pos + suffix.len(),
                    special: false,
                });
                pos += suffix.len();
                continue;
            }
            let content = if tail.starts_with(' ') && pos + 1 < end {
                pos + 1
            } else {
                pos
            };
            let class = self.class(text[content..end].chars().next().unwrap());
            let run_start = if class == 2 { pos } else { content };
            let mut run_end = run_start;
            let mut last = run_start;
            let mut count = 0;
            for (offset, ch) in text[run_start..end].char_indices() {
                if self.class(ch) != class {
                    break;
                }
                last = run_start + offset;
                run_end = last + ch.len_utf8();
                count += 1;
            }
            // Exact \s+(?!\S): before non-whitespace, leave the final whitespace
            // character for the next alternative. At EOF consume the whole run.
            if class == 2 && run_end < end && count > 1 {
                run_end = last;
            }
            out.push(Span {
                start: pos,
                end: run_end,
                special: false,
            });
            pos = run_end;
        }
    }

    /// Produce stable spans and the byte offset safe to discard from a stream.
    pub fn split(&self, text: &str, complete: bool) -> (Vec<Span>, usize) {
        let mut out = Vec::new();
        let mut segment_start = 0;
        if let Some(pattern) = &self.specials {
            for found in pattern.find_iter(text) {
                if !complete && found.start().saturating_add(self.special_width) > text.len() {
                    break;
                }
                self.ordinary(text, segment_start, found.start(), &mut out);
                out.push(Span {
                    start: found.start(),
                    end: found.end(),
                    special: true,
                });
                segment_start = found.end();
            }
        }
        let ordinary_start = out.len();
        self.ordinary(text, segment_start, text.len(), &mut out);
        if complete {
            return (out, text.len());
        }
        // Keep two trailing matches for whitespace lookahead and contractions,
        // plus enough UTF-8 bytes to cover every incomplete special marker.
        let safe_end = text
            .len()
            .saturating_sub(self.special_width.saturating_sub(1));
        let eligible_end = out.len().saturating_sub(2).max(ordinary_start);
        let keep = (ordinary_start..eligible_end)
            .find(|&i| {
                // Whitespace lookahead needs the entire next Unicode scalar,
                // while span offsets and the marker tail limit are byte offsets.
                let next_end =
                    out[i].end + text[out[i].end..].chars().next().map_or(0, char::len_utf8);
                next_end >= safe_end
            })
            .unwrap_or(eligible_end);
        out.truncate(keep);
        let consumed = out.last().map_or(segment_start, |span| span.end);
        (out, consumed)
    }
}

#[derive(Default)]
pub struct StreamState {
    pub pending: String,
}
