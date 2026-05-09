//! Mirrors `tokio::io`. We provide tokio-shaped `AsyncRead`/`AsyncWrite`
//! traits (using `ReadBuf` for reads), the extension methods mini-redis
//! exercises, and a `BufWriter` adapter.
//!
//! The shapes match tokio's deliberately so that mini-redis source can be
//! pulled in and compiled with only the import paths swapped.

use bytes::BufMut;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Tokio-shaped read buffer. Tracks initialized vs filled regions over a
/// borrowed slice. We only implement the surface used by `AsyncReadExt` here
/// — there's no `MaybeUninit` story; reads always operate on `&mut [u8]`.
pub struct ReadBuf<'a> {
    buf: &'a mut [u8],
    filled: usize,
}

impl<'a> ReadBuf<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, filled: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.filled
    }

    pub fn filled(&self) -> &[u8] {
        &self.buf[..self.filled]
    }

    pub fn filled_mut(&mut self) -> &mut [u8] {
        &mut self.buf[..self.filled]
    }

    pub fn initialize_unfilled(&mut self) -> &mut [u8] {
        &mut self.buf[self.filled..]
    }

    pub fn put_slice(&mut self, src: &[u8]) {
        let n = src.len();
        let end = self.filled + n;
        assert!(end <= self.buf.len(), "ReadBuf overflow");
        self.buf[self.filled..end].copy_from_slice(src);
        self.filled = end;
    }

    pub fn advance(&mut self, n: usize) {
        let end = self.filled + n;
        assert!(end <= self.buf.len(), "ReadBuf advance past capacity");
        self.filled = end;
    }
}

pub trait AsyncRead {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>>;
}

pub trait AsyncWrite {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>>;
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>>;
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>>;
}

// Blanket impls for &mut R / Pin<&mut R> aren't strictly needed for
// mini-redis but are common ergonomics. We add the &mut impl below.
impl<R: AsyncRead + Unpin + ?Sized> AsyncRead for &mut R {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut **self).poll_read(cx, buf)
    }
}

impl<W: AsyncWrite + Unpin + ?Sized> AsyncWrite for &mut W {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut **self).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut **self).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut **self).poll_shutdown(cx)
    }
}

// --- Extension traits (the actual surface mini-redis uses) ---

pub trait AsyncReadExt: AsyncRead {
    fn read<'a>(&'a mut self, buf: &'a mut [u8]) -> ReadFuture<'a, Self>
    where
        Self: Unpin,
    {
        ReadFuture { reader: self, buf }
    }

    fn read_exact<'a>(&'a mut self, buf: &'a mut [u8]) -> ReadExact<'a, Self>
    where
        Self: Unpin,
    {
        ReadExact {
            reader: self,
            buf,
            pos: 0,
        }
    }

    /// Read into a `BufMut`, returning the number of bytes read.
    /// Behaves like tokio's `read_buf`: writes into the unused capacity of
    /// the buffer and advances its length.
    fn read_buf<'a, B: BufMut>(&'a mut self, buf: &'a mut B) -> ReadBufFuture<'a, Self, B>
    where
        Self: Unpin,
    {
        ReadBufFuture { reader: self, buf }
    }
}

impl<R: AsyncRead + ?Sized> AsyncReadExt for R {}

pub struct ReadFuture<'a, R: ?Sized> {
    reader: &'a mut R,
    buf: &'a mut [u8],
}

impl<'a, R: AsyncRead + Unpin + ?Sized> Future for ReadFuture<'a, R> {
    type Output = io::Result<usize>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();
        let mut rb = ReadBuf::new(me.buf);
        match Pin::new(&mut *me.reader).poll_read(cx, &mut rb) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Ready(Ok(())) => Poll::Ready(Ok(rb.filled().len())),
        }
    }
}

pub struct ReadExact<'a, R: ?Sized> {
    reader: &'a mut R,
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a, R: AsyncRead + Unpin + ?Sized> Future for ReadExact<'a, R> {
    type Output = io::Result<usize>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();
        loop {
            if me.pos == me.buf.len() {
                return Poll::Ready(Ok(me.pos));
            }
            let mut rb = ReadBuf::new(&mut me.buf[me.pos..]);
            match Pin::new(&mut *me.reader).poll_read(cx, &mut rb) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(())) => {
                    let n = rb.filled().len();
                    if n == 0 {
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "early eof",
                        )));
                    }
                    me.pos += n;
                }
            }
        }
    }
}

pub struct ReadBufFuture<'a, R: ?Sized, B: BufMut> {
    reader: &'a mut R,
    buf: &'a mut B,
}

impl<'a, R: AsyncRead + Unpin + ?Sized, B: BufMut> Future for ReadBufFuture<'a, R, B> {
    type Output = io::Result<usize>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();
        // Use an intermediate stack buffer. We can't safely hand a slice of
        // unused BufMut capacity to poll_read because BufMut::chunk_mut
        // returns &mut UninitSlice, and our ReadBuf only knows &mut [u8].
        //
        // Use a 4 KiB temp buffer; for mini-redis's 4 KiB read buffer this
        // matches behavior reasonably (one syscall per pump). Real tokio
        // does it without the copy via MaybeUninit; we accept the copy for
        // shim simplicity.
        let mut tmp = [0u8; 4096];
        let max = me.buf.remaining_mut().min(tmp.len());
        if max == 0 {
            return Poll::Ready(Ok(0));
        }
        let mut rb = ReadBuf::new(&mut tmp[..max]);
        match Pin::new(&mut *me.reader).poll_read(cx, &mut rb) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Ready(Ok(())) => {
                let n = rb.filled().len();
                if n > 0 {
                    me.buf.put_slice(&tmp[..n]);
                }
                Poll::Ready(Ok(n))
            }
        }
    }
}

// --- Write extension ---

pub trait AsyncWriteExt: AsyncWrite {
    fn write<'a>(&'a mut self, buf: &'a [u8]) -> WriteFuture<'a, Self>
    where
        Self: Unpin,
    {
        WriteFuture { writer: self, buf }
    }

    fn write_all<'a>(&'a mut self, buf: &'a [u8]) -> WriteAllFuture<'a, Self>
    where
        Self: Unpin,
    {
        WriteAllFuture {
            writer: self,
            buf,
            pos: 0,
        }
    }

    fn write_u8<'a>(&'a mut self, n: u8) -> WriteAllOwnedFuture<'a, Self, [u8; 1]>
    where
        Self: Unpin,
    {
        WriteAllOwnedFuture {
            writer: self,
            buf: [n],
            pos: 0,
        }
    }

    fn flush(&mut self) -> FlushFuture<'_, Self>
    where
        Self: Unpin,
    {
        FlushFuture { writer: self }
    }

    fn shutdown(&mut self) -> ShutdownFuture<'_, Self>
    where
        Self: Unpin,
    {
        ShutdownFuture { writer: self }
    }
}

impl<W: AsyncWrite + ?Sized> AsyncWriteExt for W {}

pub struct WriteFuture<'a, W: ?Sized> {
    writer: &'a mut W,
    buf: &'a [u8],
}

impl<'a, W: AsyncWrite + Unpin + ?Sized> Future for WriteFuture<'a, W> {
    type Output = io::Result<usize>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();
        Pin::new(&mut *me.writer).poll_write(cx, me.buf)
    }
}

pub struct WriteAllFuture<'a, W: ?Sized> {
    writer: &'a mut W,
    buf: &'a [u8],
    pos: usize,
}

impl<'a, W: AsyncWrite + Unpin + ?Sized> Future for WriteAllFuture<'a, W> {
    type Output = io::Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();
        while me.pos < me.buf.len() {
            match Pin::new(&mut *me.writer).poll_write(cx, &me.buf[me.pos..]) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "wrote zero bytes",
                    )))
                }
                Poll::Ready(Ok(n)) => me.pos += n,
            }
        }
        Poll::Ready(Ok(()))
    }
}

pub struct WriteAllOwnedFuture<'a, W: ?Sized, B> {
    writer: &'a mut W,
    buf: B,
    pos: usize,
}

impl<'a, W: AsyncWrite + Unpin + ?Sized, B: AsRef<[u8]> + Unpin> Future
    for WriteAllOwnedFuture<'a, W, B>
{
    type Output = io::Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();
        let bytes = me.buf.as_ref();
        while me.pos < bytes.len() {
            match Pin::new(&mut *me.writer).poll_write(cx, &bytes[me.pos..]) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "wrote zero bytes",
                    )))
                }
                Poll::Ready(Ok(n)) => me.pos += n,
            }
        }
        Poll::Ready(Ok(()))
    }
}

pub struct FlushFuture<'a, W: ?Sized> {
    writer: &'a mut W,
}

impl<'a, W: AsyncWrite + Unpin + ?Sized> Future for FlushFuture<'a, W> {
    type Output = io::Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();
        Pin::new(&mut *me.writer).poll_flush(cx)
    }
}

pub struct ShutdownFuture<'a, W: ?Sized> {
    writer: &'a mut W,
}

impl<'a, W: AsyncWrite + Unpin + ?Sized> Future for ShutdownFuture<'a, W> {
    type Output = io::Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();
        Pin::new(&mut *me.writer).poll_shutdown(cx)
    }
}

// --- BufWriter ---

/// Buffered writer that batches small writes. Tokio's `BufWriter` is used by
/// mini-redis's `Connection` to avoid one syscall per byte. Our impl is a
/// simple ring-free buffer that flushes when full.
pub struct BufWriter<W> {
    inner: W,
    buf: Vec<u8>,
    cap: usize,
    /// Position of the next byte to emit during a flush — used so that when
    /// `poll_write` of the inner writer returns Pending mid-flush, we can
    /// resume.
    flushed: usize,
}

impl<W: std::fmt::Debug> std::fmt::Debug for BufWriter<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufWriter")
            .field("inner", &self.inner)
            .field("buffered", &self.buf.len())
            .finish()
    }
}

impl<W> BufWriter<W> {
    pub fn new(inner: W) -> Self {
        Self::with_capacity(8 * 1024, inner)
    }

    pub fn with_capacity(cap: usize, inner: W) -> Self {
        Self {
            inner,
            buf: Vec::with_capacity(cap),
            cap,
            flushed: 0,
        }
    }

    pub fn get_ref(&self) -> &W {
        &self.inner
    }
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: AsyncWrite + Unpin> BufWriter<W> {
    /// Drain the buffer into the inner writer. Used by poll_flush. Returns
    /// Pending if the inner writer can't accept right now.
    fn poll_flush_buf(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        while self.flushed < self.buf.len() {
            match Pin::new(&mut self.inner).poll_write(cx, &self.buf[self.flushed..]) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "wrote zero bytes (BufWriter flush)",
                    )))
                }
                Poll::Ready(Ok(n)) => self.flushed += n,
            }
        }
        // Buffer drained; reset.
        self.buf.clear();
        self.flushed = 0;
        Poll::Ready(Ok(()))
    }
}

impl<W: AsyncWrite + Unpin> AsyncWrite for BufWriter<W> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        src: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        // If incoming write is larger than capacity, flush first then write
        // direct.
        if src.len() >= me.cap {
            match me.poll_flush_buf(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(())) => {}
            }
            return Pin::new(&mut me.inner).poll_write(cx, src);
        }
        // If buffer would overflow, flush first.
        if me.buf.len() + src.len() > me.cap {
            match me.poll_flush_buf(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(())) => {}
            }
        }
        me.buf.extend_from_slice(src);
        Poll::Ready(Ok(src.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        match me.poll_flush_buf(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Ready(Ok(())) => Pin::new(&mut me.inner).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        match me.poll_flush_buf(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Ready(Ok(())) => Pin::new(&mut me.inner).poll_shutdown(cx),
        }
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for BufWriter<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        Pin::new(&mut me.inner).poll_read(cx, buf)
    }
}
