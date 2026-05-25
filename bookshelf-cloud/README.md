# bookshelf-cloud

Static-hosted AudiobookShelf mock for BookPlayer iOS. See [DESIGN.md](DESIGN.md).

## Layout

```
bookshelf-cloud/
├── DESIGN.md
├── cdk/                  # CDK TypeScript app — S3, CloudFront, Lambda, secret
└── publish/              # Python CLI — uploads m4bs + updates manifest JSON
```

## First-time deploy

Prereqs: AWS CLI configured (`aws sts get-caller-identity` works), Node 22+,
Python 3.11+, `uv` installed.

### 1. Generate the JWT secret and a bcrypt-hashed password

```sh
cd cdk
JWT_SECRET=$(openssl rand -hex 32)
PASSWORD='choose-a-real-password-here'
BCRYPT_HASH=$(node -e "console.log(require('bcryptjs').hashSync(process.argv[1], 12))" "$PASSWORD")
echo "JWT_SECRET=$JWT_SECRET"
echo "BCRYPT_HASH=$BCRYPT_HASH"
```

Save both. The JWT secret is needed for every redeploy (it's baked into the
CloudFront Function source). The bcrypt hash only matters for `cdk deploy`.

### 2. Bootstrap CDK in your account (one-time per region)

```sh
npx cdk bootstrap
```

### 3. Deploy

```sh
npx cdk deploy \
  -c jwtSecret=$JWT_SECRET \
  -c username=admin \
  -c bcryptHash="$BCRYPT_HASH"
```

CDK prints three outputs at the end:

- `BucketName` — copy into `~/.config/bookshelf-cloud/config.json`
- `DistributionDomain` — paste into BookPlayer as the server URL
- `CredentialsSecretArn` — for future password rotation

### 4. Configure the publish CLI

```sh
mkdir -p ~/.config/bookshelf-cloud
cat > ~/.config/bookshelf-cloud/config.json <<EOF
{"bucket": "<paste BucketName here>"}
EOF
```

### 5. Bootstrap the static endpoints

```sh
cd ../publish
uv run bookshelf-cloud bootstrap
```

This writes the static JSON files (`/ping`, `/api/libraries`, etc) that don't
depend on any book.

### 6. Add BookPlayer

In BookPlayer (iOS): Settings → AudiobookShelf → Add Server.

- Server URL: the `DistributionDomain` value
- Username: `admin`
- Password: whatever you used in step 1

## Publishing books

The watcher (`~/audiobooks/bin/watcher.py`) now auto-publishes each finished
m4b. Drop a PDF in `~/audiobooks/inbox/`, wait for the pipeline, and it
appears in BookPlayer.

To publish manually:

```sh
cd publish
uv run bookshelf-cloud publish ~/audiobooks/m4b/some-book.m4b
uv run bookshelf-cloud list
uv run bookshelf-cloud remove some-book
```

## Rotating credentials

Bump the password (new bcrypt hash):

```sh
aws secretsmanager update-secret \
  --secret-id bookshelf-cloud/credentials \
  --secret-string "$(jq -n --arg j "$JWT_SECRET" --arg u admin --arg h "$NEW_HASH" \
    '{jwtSecret:$j, username:$u, bcryptHash:$h}')"
```

Bump the JWT secret (invalidates all outstanding tokens; users must log in
again from BookPlayer):

```sh
npx cdk deploy -c jwtSecret=$NEW_JWT -c username=admin -c bcryptHash=$BCRYPT_HASH
```

Then update Secrets Manager with the new `jwtSecret` value so the login
Lambda signs tokens that the new CF Function accepts.
