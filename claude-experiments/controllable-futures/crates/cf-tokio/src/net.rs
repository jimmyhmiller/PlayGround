//! Mirrors `tokio::net`. Wraps `async-net`'s `TcpListener`/`TcpStream` with
//! tokio-shaped APIs (accept returns `(TcpStream, SocketAddr)`, bind takes
//! `ToSocketAddrs`, our streams impl our `AsyncRead`/`AsyncWrite`).

use crate::io::{AsyncRead, AsyncWrite, ReadBuf};
use cf_runtime::resource::{
    ResourceId, ResourceKind, ResourceProbe, ResourceRegistry, ResourceStateSnapshot,
};
use futures_lite::{AsyncRead as FlAsyncRead, AsyncWrite as FlAsyncWrite};
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

pub use std::net::ToSocketAddrs;

struct ListenerProbe {
    inner: async_net::TcpListener,
}
impl ResourceProbe for ListenerProbe {
    fn snapshot(&self) -> ResourceStateSnapshot {
        let local = self.inner.local_addr().map(|a| a.to_string()).ok();
        ResourceStateSnapshot {
            local,
            ..Default::default()
        }
    }
}

struct StreamProbe {
    peer: String,
    bytes_read: Arc<std::sync::atomic::AtomicU64>,
    bytes_written: Arc<std::sync::atomic::AtomicU64>,
}
impl ResourceProbe for StreamProbe {
    fn snapshot(&self) -> ResourceStateSnapshot {
        ResourceStateSnapshot {
            peer: Some(self.peer.clone()),
            sends: self
                .bytes_written
                .load(std::sync::atomic::Ordering::Relaxed),
            recvs: self.bytes_read.load(std::sync::atomic::Ordering::Relaxed),
            ..Default::default()
        }
    }
}

pub struct TcpListener {
    inner: async_net::TcpListener,
    resource_id: Option<ResourceId>,
    registry: Option<Arc<ResourceRegistry>>,
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        if let (Some(id), Some(reg)) = (self.resource_id.take(), self.registry.take()) {
            reg.remove(id);
        }
    }
}

impl std::fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpListener").finish()
    }
}

impl TcpListener {
    pub async fn bind<A: ToSocketAddrs>(addrs: A) -> io::Result<Self> {
        let resolved: Vec<SocketAddr> = addrs.to_socket_addrs()?.collect();
        let inner = async_net::TcpListener::bind(&resolved[..]).await?;
        let local = inner.local_addr().map(|a| a.to_string()).unwrap_or_default();
        let registry = cf_runtime::try_current().map(|h| h.resources());
        let resource_id = registry.as_ref().map(|reg| {
            reg.insert(
                ResourceKind::TcpListener,
                format!("listen {local}"),
                cf_runtime::runtime::current_task(),
                Arc::new(ListenerProbe {
                    inner: inner.clone(),
                }),
            )
        });
        Ok(Self {
            inner,
            resource_id,
            registry,
        })
    }

    pub async fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        let local = self
            .inner
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "?".into());
        crate::note_wait(cf_runtime::WaitReason::TcpAccept { local_addr: local });
        let (s, addr) = self.inner.accept().await?;
        let peer = addr.to_string();
        if let Some(h) = cf_runtime::try_current() {
            h.log_user_event("net", format!("accept {}", addr));
            h.clear_wait_reason();
        }
        let (br, bw, id, reg) = register_stream(&peer);
        Ok((
            TcpStream {
                inner: s,
                peer,
                bytes_read: br,
                bytes_written: bw,
                resource_id: id,
                registry: reg,
            },
            addr,
        ))
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

pub struct TcpStream {
    inner: async_net::TcpStream,
    /// Stored peer address for wait-reason labels. We capture it at
    /// accept/connect time because re-querying via `peer_addr` allocates
    /// and can fail after socket close.
    peer: String,
    bytes_read: Arc<std::sync::atomic::AtomicU64>,
    bytes_written: Arc<std::sync::atomic::AtomicU64>,
    resource_id: Option<ResourceId>,
    registry: Option<Arc<ResourceRegistry>>,
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        if let (Some(id), Some(reg)) = (self.resource_id.take(), self.registry.take()) {
            reg.remove(id);
        }
    }
}

fn register_stream(peer: &str) -> (
    Arc<std::sync::atomic::AtomicU64>,
    Arc<std::sync::atomic::AtomicU64>,
    Option<ResourceId>,
    Option<Arc<ResourceRegistry>>,
) {
    let bytes_read = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let bytes_written = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let registry = cf_runtime::try_current().map(|h| h.resources());
    let resource_id = registry.as_ref().map(|reg| {
        reg.insert(
            ResourceKind::TcpStream,
            format!("stream {peer}"),
            cf_runtime::runtime::current_task(),
            Arc::new(StreamProbe {
                peer: peer.to_string(),
                bytes_read: bytes_read.clone(),
                bytes_written: bytes_written.clone(),
            }),
        )
    });
    (bytes_read, bytes_written, resource_id, registry)
}

impl std::fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpStream").finish()
    }
}

impl TcpStream {
    pub async fn connect<A: ToSocketAddrs>(addrs: A) -> io::Result<Self> {
        let resolved: Vec<SocketAddr> = addrs.to_socket_addrs()?.collect();
        let inner = async_net::TcpStream::connect(&resolved[..]).await?;
        let peer = inner
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "?".into());
        let (br, bw, id, reg) = register_stream(&peer);
        Ok(Self {
            inner,
            peer,
            bytes_read: br,
            bytes_written: bw,
            resource_id: id,
            registry: reg,
        })
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }
}

impl AsyncRead for TcpStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        let dst = buf.initialize_unfilled();
        match Pin::new(&mut me.inner).poll_read(cx, dst) {
            Poll::Pending => {
                crate::note_wait(cf_runtime::WaitReason::TcpRead {
                    peer: me.peer.clone(),
                });
                Poll::Pending
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Ready(Ok(n)) => {
                me.bytes_read
                    .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                buf.advance(n);
                Poll::Ready(Ok(()))
            }
        }
    }
}

impl AsyncWrite for TcpStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        match Pin::new(&mut me.inner).poll_write(cx, buf) {
            Poll::Pending => {
                crate::note_wait(cf_runtime::WaitReason::TcpWrite {
                    peer: me.peer.clone(),
                });
                Poll::Pending
            }
            Poll::Ready(Ok(n)) => {
                me.bytes_written
                    .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                Poll::Ready(Ok(n))
            }
            r => r,
        }
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        match Pin::new(&mut me.inner).poll_flush(cx) {
            Poll::Pending => {
                crate::note_wait(cf_runtime::WaitReason::TcpFlush {
                    peer: me.peer.clone(),
                });
                Poll::Pending
            }
            r => r,
        }
    }
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        Pin::new(&mut me.inner).poll_close(cx)
    }
}
