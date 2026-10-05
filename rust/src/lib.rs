//! Whole-text FFI, native streaming boundaries, and packed token blocks.

mod bpe;
mod encoding;
mod pretokenization;

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyList};

use bpe::Encoder;
use pretokenization::{PreTokenizer, StreamState};

struct Model {
    encoder: Encoder,
    pretokenizer: PreTokenizer,
    // Keep arbitrary Python integer IDs for the public list interface.
    token_ids: Vec<Py<PyInt>>,
}

impl Model {
    fn list(&self, py: Python<'_>, indices: &[usize]) -> PyResult<Py<PyList>> {
        Ok(PyList::new(py, indices.iter().map(|&i| self.token_ids[i].bind(py)))?.unbind())
    }

    fn push(
        &self,
        state: &mut StreamState,
        text: &str,
        final_chunk: bool,
    ) -> Result<Vec<usize>, Vec<u8>> {
        state.pending.push_str(text);
        let (spans, consumed) = self.pretokenizer.split(&state.pending, final_chunk);
        let ids = self.encoder.encode_spans(&state.pending, &spans)?;
        state.pending.drain(..consumed);
        Ok(ids)
    }
}

/// A frozen model shared by independently owned stream states.
#[pyclass(frozen, module = "_lmfs_bpe")]
struct NativeBPE {
    model: Arc<Model>,
}

#[pymethods]
impl NativeBPE {
    #[new]
    fn new(
        token_to_id: HashMap<Vec<u8>, Py<PyInt>>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Vec<String>,
    ) -> Self {
        let mut token_ids = Vec::with_capacity(token_to_id.len());
        let mut token_to_index = HashMap::with_capacity(token_to_id.len());
        for (token, token_id) in token_to_id {
            token_to_index.insert(token, token_ids.len());
            token_ids.push(token_id);
        }
        Self {
            model: Arc::new(Model {
                encoder: Encoder::new(token_to_index, merges),
                pretokenizer: PreTokenizer::new(special_tokens),
                token_ids,
            }),
        }
    }

    fn encode_text(&self, py: Python<'_>, text: String) -> PyResult<Py<PyList>> {
        // Keep pre-token spans and intermediate ID arrays bounded even when
        // the caller supplies a huge string. The returned list still owns all IDs.
        let output = PyList::empty(py);
        let mut state = StreamState::default();
        let mut start = 0;
        while start < text.len() {
            let mut end = (start + 65536).min(text.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            let indices = py
                .detach(|| {
                    self.model
                        .push(&mut state, &text[start..end], end == text.len())
                })
                .map_err(|missing| PyKeyError::new_err((missing,)))?;
            let batch = self.model.list(py, &indices)?;
            output.call_method1("extend", (batch,))?;
            start = end;
        }
        Ok(output.unbind())
    }

    #[pyo3(signature = (dtype=None))]
    fn stream(&self, py: Python<'_>, dtype: Option<&str>) -> PyResult<NativeStream> {
        let itemsize = match dtype {
            None => None,
            Some("uint16" | "<u2") => Some(2),
            Some("uint32" | "<u4") => Some(4),
            Some(_) => {
                return Err(PyValueError::new_err(
                    "dtype must be uint16, uint32, <u2 or <u4",
                ));
            }
        };
        let packed_ids = if let Some(width) = itemsize {
            let max = if width == 2 {
                u16::MAX as u32
            } else {
                u32::MAX
            };
            // A narrowing cast would silently change the packed token IDs.
            Some(PackedIds {
                width,
                table: self
                    .model
                    .token_ids
                    .iter()
                    .map(|id| {
                        id.extract::<u32>(py)
                            .ok()
                            .filter(|&id| id <= max)
                            .ok_or_else(|| {
                                PyValueError::new_err(
                                    "vocabulary IDs must fit the requested unsigned dtype",
                                )
                            })
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            })
        } else {
            None
        };
        Ok(NativeStream {
            model: self.model.clone(),
            state: StreamState::default(),
            packed_ids,
        })
    }
}

/// One continuous input stream; safe prefixes are encoded and tails retained.
#[pyclass(module = "_lmfs_bpe")]
struct NativeStream {
    model: Arc<Model>,
    state: StreamState,
    packed_ids: Option<PackedIds>,
}

struct PackedIds {
    width: usize,
    table: Vec<u32>,
}

#[pymethods]
impl NativeStream {
    #[pyo3(signature = (text, final_chunk=false))]
    fn push(&mut self, py: Python<'_>, text: String, final_chunk: bool) -> PyResult<Py<PyList>> {
        let indices = py
            .detach(|| self.model.push(&mut self.state, &text, final_chunk))
            .map_err(|missing| PyKeyError::new_err((missing,)))?;
        self.model.list(py, &indices)
    }

    #[pyo3(signature = (text, final_chunk=false))]
    fn push_bytes(
        &mut self,
        py: Python<'_>,
        text: String,
        final_chunk: bool,
    ) -> PyResult<Py<PyBytes>> {
        let packed = self
            .packed_ids
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("create a packed stream with dtype"))?;
        let bytes = py
            .detach(|| {
                let indices = self.model.push(&mut self.state, &text, final_chunk)?;
                let mut bytes = Vec::with_capacity(indices.len() * packed.width);
                for index in indices {
                    let id = packed.table[index];
                    if packed.width == 2 {
                        bytes.extend_from_slice(&(id as u16).to_le_bytes());
                    } else {
                        bytes.extend_from_slice(&id.to_le_bytes());
                    }
                }
                Ok::<_, Vec<u8>>(bytes)
            })
            .map_err(|missing| PyKeyError::new_err((missing,)))?;
        // One bounded bulk copy; no per-token Python objects. Python bytes owns
        // the immutable buffer, so file writes and np.frombuffer can borrow it.
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

#[pymodule]
fn _lmfs_bpe(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let (major, minor, patch) = unicode_properties::UNICODE_VERSION;
    module.add("UNICODE_VERSION", format!("{major}.{minor}.{patch}"))?;
    module.add_class::<NativeBPE>()?;
    module.add_class::<NativeStream>()
}
