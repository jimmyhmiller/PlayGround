// bookshelf-cloud — single-Lambda handler.
//
// Routes every request from BookPlayer:
//   GET  /ping                                  -> {success: true}
//   POST /login                                 -> mint JWT (rate-limited via S3 counter)
//   GET  /api/libraries                         -> manifest JSON
//   GET  /api/libraries/{libId}/items           -> manifest JSON
//   GET  /api/libraries/{libId}/filterdata      -> manifest JSON
//   GET  /api/libraries/{libId}/collections     -> manifest JSON
//   GET  /api/libraries/{libId}/search          -> manifest JSON
//   GET  /api/items/{id}                        -> manifest JSON
//   GET  /api/items/{id}/cover                  -> 302 to presigned S3 URL
//   GET  /api/items/{id}/download               -> 302 to presigned S3 URL
//
// Auth: every endpoint except /login and /ping requires a valid JWT in
// `Authorization: Bearer <jwt>` OR `?token=<jwt>` (the download URL uses
// the query-param form because AVPlayer does not forward custom headers).

import { SecretsManagerClient, GetSecretValueCommand } from '@aws-sdk/client-secrets-manager';
import { S3Client, GetObjectCommand, PutObjectCommand, NoSuchKey } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import bcrypt from 'bcryptjs';
import crypto from 'node:crypto';

const sm = new SecretsManagerClient({});
const s3 = new S3Client({});

const BUCKET = process.env.BUCKET_NAME;
const JWT_TTL_SECONDS = parseInt(process.env.JWT_TTL_SECONDS, 10);
const PRESIGN_TTL_SECONDS = 30 * 60;

const WINDOW_SECONDS = 5 * 60;
const MAX_ATTEMPTS = 10;
const CAS_MAX_RETRIES = 4;

let credsCache = null;
let credsCachedAt = 0;
const CREDS_CACHE_TTL_MS = 60_000;

async function loadCredentials() {
    const now = Date.now();
    if (credsCache && now - credsCachedAt < CREDS_CACHE_TTL_MS) return credsCache;
    const out = await sm.send(new GetSecretValueCommand({
        SecretId: process.env.CREDENTIALS_SECRET_ARN,
    }));
    credsCache = JSON.parse(out.SecretString);
    credsCachedAt = now;
    return credsCache;
}

function jsonResponse(statusCode, body, extraHeaders = {}) {
    return {
        statusCode,
        headers: { 'content-type': 'application/json', ...extraHeaders },
        body: JSON.stringify(body),
    };
}

function redirect(location) {
    return {
        statusCode: 302,
        headers: { location, 'cache-control': 'no-store' },
        body: '',
    };
}

function b64url(buf) {
    return buf.toString('base64')
        .replace(/=/g, '')
        .replace(/\+/g, '-')
        .replace(/\//g, '_');
}

function sign(payload, secret) {
    const header = b64url(Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })));
    const body = b64url(Buffer.from(JSON.stringify(payload)));
    const signingInput = `${header}.${body}`;
    const sig = b64url(
        crypto.createHmac('sha256', secret).update(signingInput).digest()
    );
    return `${signingInput}.${sig}`;
}

function verifyJwt(token, secret) {
    if (!token) return null;
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const signingInput = parts[0] + '.' + parts[1];
    const expected = crypto.createHmac('sha256', secret).update(signingInput).digest();
    let provided;
    try {
        provided = Buffer.from(parts[2], 'base64url');
    } catch {
        return null;
    }
    if (expected.length !== provided.length) return null;
    if (!crypto.timingSafeEqual(expected, provided)) return null;
    let claims;
    try {
        claims = JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf8'));
    } catch {
        return null;
    }
    const now = Math.floor(Date.now() / 1000);
    if (typeof claims.exp !== 'number' || claims.exp < now) return null;
    return claims;
}

function extractToken(event) {
    const headers = event.headers || {};
    const auth = headers['authorization'] || headers['Authorization'];
    if (auth && auth.startsWith('Bearer ')) return auth.slice(7);
    const qs = event.queryStringParameters;
    if (qs && qs.token) return qs.token;
    return null;
}

async function readCounter(key) {
    try {
        const out = await s3.send(new GetObjectCommand({ Bucket: BUCKET, Key: key }));
        const body = await out.Body.transformToString('utf8');
        return { state: JSON.parse(body), etag: out.ETag };
    } catch (e) {
        if (e instanceof NoSuchKey || e?.name === 'NoSuchKey') {
            return { state: null, etag: null };
        }
        throw e;
    }
}

async function writeCounter(key, state, ifMatch) {
    const params = {
        Bucket: BUCKET,
        Key: key,
        Body: JSON.stringify(state),
        ContentType: 'application/json',
    };
    if (ifMatch) params.IfMatch = ifMatch;
    else params.IfNoneMatch = '*';
    await s3.send(new PutObjectCommand(params));
}

async function checkAndIncrement(ip) {
    const key = `ratelimit/${ip}.json`;
    const nowSec = Math.floor(Date.now() / 1000);
    for (let i = 0; i < CAS_MAX_RETRIES; i++) {
        const { state, etag } = await readCounter(key);
        let next;
        if (!state || nowSec - (state.windowStart || 0) >= WINDOW_SECONDS) {
            next = { windowStart: nowSec, count: 1 };
        } else {
            if (state.count >= MAX_ATTEMPTS) {
                return { allowed: false, retryAfter: WINDOW_SECONDS - (nowSec - state.windowStart) };
            }
            next = { windowStart: state.windowStart, count: state.count + 1 };
        }
        try {
            await writeCounter(key, next, etag);
            return { allowed: true };
        } catch (e) {
            const code = e?.$metadata?.httpStatusCode;
            if (code === 412 || code === 409 || e?.name === 'PreconditionFailed') continue;
            throw e;
        }
    }
    return { allowed: false, retryAfter: WINDOW_SECONDS };
}

function clientIp(event) {
    const xff = event.headers?.['x-forwarded-for'] || event.headers?.['X-Forwarded-For'];
    if (xff) {
        const first = xff.split(',')[0].trim();
        if (first) return first;
    }
    return event.requestContext?.http?.sourceIp || 'unknown';
}

async function serveManifest(key) {
    try {
        const out = await s3.send(new GetObjectCommand({ Bucket: BUCKET, Key: key }));
        const body = await out.Body.transformToString('utf8');
        return {
            statusCode: 200,
            headers: { 'content-type': 'application/json' },
            body,
        };
    } catch (e) {
        if (e instanceof NoSuchKey || e?.name === 'NoSuchKey') {
            return jsonResponse(404, { error: 'not found', key });
        }
        throw e;
    }
}

async function presign(key) {
    const cmd = new GetObjectCommand({ Bucket: BUCKET, Key: key });
    return getSignedUrl(s3, cmd, { expiresIn: PRESIGN_TTL_SECONDS });
}

async function handleLogin(event) {
    if (event.requestContext?.http?.method !== 'POST') {
        return jsonResponse(405, { error: 'method not allowed' });
    }

    const ip = clientIp(event);
    let rl;
    try {
        rl = await checkAndIncrement(ip);
    } catch (e) {
        console.log('rate limiter error: ' + (e?.message || e));
        return jsonResponse(503, { error: 'service unavailable' });
    }
    if (!rl.allowed) {
        return jsonResponse(429, { error: 'too many attempts, try again later' },
            { 'retry-after': String(rl.retryAfter) });
    }

    let username, password;
    try {
        const body = event.isBase64Encoded
            ? Buffer.from(event.body, 'base64').toString('utf8')
            : (event.body || '');
        const parsed = JSON.parse(body || '{}');
        username = parsed.username;
        password = parsed.password;
    } catch {
        return jsonResponse(400, { error: 'invalid json' });
    }
    if (!username || !password) {
        return jsonResponse(400, { error: 'username and password required' });
    }

    const creds = await loadCredentials();
    if (username !== creds.username) {
        return jsonResponse(401, { error: 'invalid credentials' });
    }
    console.log(`pw length=${password.length}`);
    const ok = await bcrypt.compare(password, creds.bcryptHash);
    if (!ok) {
        return jsonResponse(401, { error: 'invalid credentials' });
    }

    const now = Math.floor(Date.now() / 1000);
    const token = sign(
        { sub: creds.username, iat: now, exp: now + JWT_TTL_SECONDS },
        creds.jwtSecret,
    );

    s3.send(new PutObjectCommand({
        Bucket: BUCKET,
        Key: `ratelimit/${ip}.json`,
        Body: JSON.stringify({ windowStart: now, count: 0 }),
        ContentType: 'application/json',
    })).catch(() => {});

    return jsonResponse(200, {
        user: { id: creds.username, token },
        serverSettings: {},
        libraries: [],
    });
}

async function handleAuthed(event, path) {
    const creds = await loadCredentials();
    if (!verifyJwt(extractToken(event), creds.jwtSecret)) {
        return jsonResponse(401, { error: 'invalid or missing token' });
    }

    const mediaMatch = path.match(/^\/api\/items\/([^\/]+)\/(cover|download)$/);
    if (mediaMatch) {
        const id = mediaMatch[1];
        const key = mediaMatch[2] === 'cover'
            ? `books/${id}/cover.jpg`
            : `books/${id}/audio.m4b`;
        const url = await presign(key);
        return redirect(url);
    }

    if (path.startsWith('/api/')) {
        const key = path.slice(1);
        return serveManifest(key);
    }

    return jsonResponse(404, { error: 'not found' });
}

async function logged(event, h) {
    const r = await h();
    console.log(`RESP ${r.statusCode}`);
    return r;
}

export async function handler(event) {
    const method = event.requestContext?.http?.method;
    const path = event.rawPath || '/';
    const ua = event.headers?.['user-agent'] || '';
    console.log(`REQ ${method} ${path} ua=${ua} qs=${event.rawQueryString || ''}`);
    console.log(`REQ headers: ${JSON.stringify(event.headers || {})}`);
    if (event.body) {
        const body = event.isBase64Encoded
            ? Buffer.from(event.body, 'base64').toString('utf8')
            : event.body;
        const safe = body.replace(/"password"\s*:\s*"[^"]*"/, '"password":"<redacted>"');
        console.log(`REQ body: ${safe}`);
    }

    try {
        let resp;
        if (path === '/ping') {
            if (method !== 'GET') resp = jsonResponse(405, { error: 'method not allowed' });
            else resp = jsonResponse(200, { success: true });
        } else if (path === '/login') {
            resp = await handleLogin(event);
        } else if (path.startsWith('/api/')) {
            if (method !== 'GET') resp = jsonResponse(405, { error: 'method not allowed' });
            else resp = await handleAuthed(event, path);
        } else {
            resp = jsonResponse(404, { error: 'not found', path });
        }
        console.log(`RESP ${resp.statusCode} body=${typeof resp.body === 'string' ? resp.body.slice(0, 200) : ''}`);
        return resp;
    } catch (e) {
        console.log('handler error: ' + (e?.stack || e?.message || e));
        return jsonResponse(500, { error: 'internal' });
    }
}
