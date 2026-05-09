//! STUB: the original mini-redis SUBSCRIBE/UNSUBSCRIBE implementation
//! depends on `tokio_stream::StreamMap`, which we haven't ported into the
//! cf-tokio shim. We keep `parse_frames` working so the protocol isn't
//! broken, but `apply` returns an explicit error so any client attempting
//! to subscribe gets a clear rejection rather than a silent hang.
//!
//! Replace this stub with the original implementation once the shim grows
//! a `tokio_stream` analog.

use crate::cmd::{Parse, ParseError};
use crate::{Connection, Db, Frame, Shutdown};

#[derive(Debug)]
pub struct Subscribe {
    channels: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Unsubscribe {
    channels: Vec<String>,
}

impl Subscribe {
    pub(crate) fn new(channels: Vec<String>) -> Self {
        Self { channels }
    }

    pub(crate) fn parse_frames(parse: &mut Parse) -> crate::Result<Self> {
        let mut channels = vec![parse.next_string()?];
        loop {
            match parse.next_string() {
                Ok(s) => channels.push(s),
                Err(ParseError::EndOfStream) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(Self { channels })
    }

    pub(crate) async fn apply(
        self,
        _db: &Db,
        _dst: &mut Connection,
        _shutdown: &mut Shutdown,
    ) -> crate::Result<()> {
        Err(format!(
            "SUBSCRIBE not supported in cf-tokio mini-redis (stream shim pending); channels: {:?}",
            self.channels
        )
        .into())
    }

    pub(crate) fn into_frame(self) -> Frame {
        let mut frame = Frame::array();
        frame.push_bulk(bytes::Bytes::from("subscribe".as_bytes()));
        for ch in self.channels {
            frame.push_bulk(bytes::Bytes::from(ch.into_bytes()));
        }
        frame
    }
}

impl Unsubscribe {
    pub(crate) fn new(channels: &[String]) -> Self {
        Self {
            channels: channels.to_vec(),
        }
    }

    pub(crate) fn parse_frames(parse: &mut Parse) -> Result<Self, ParseError> {
        let mut channels = Vec::new();
        loop {
            match parse.next_string() {
                Ok(s) => channels.push(s),
                Err(ParseError::EndOfStream) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(Self { channels })
    }

    pub(crate) fn into_frame(self) -> Frame {
        let mut frame = Frame::array();
        frame.push_bulk(bytes::Bytes::from("unsubscribe".as_bytes()));
        for ch in self.channels {
            frame.push_bulk(bytes::Bytes::from(ch.into_bytes()));
        }
        frame
    }
}
