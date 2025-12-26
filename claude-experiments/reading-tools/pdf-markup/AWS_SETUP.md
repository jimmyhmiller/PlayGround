# AWS SDK Setup Guide for PDF Markup S3 Sync

This guide explains how to enable S3 cloud sync for PDF drawings in the PDF Markup app.

## Current Status

✅ **Phase 1 & 2 Complete**: Local persistence and export functionality work without AWS SDK
⚠️ **Phase 3 Ready**: S3 sync infrastructure is in place but requires AWS SDK to be activated

## What Works Now (Without AWS SDK)

- ✅ Drawings persist locally across app restarts
- ✅ Export PDFs with markups baked in
- ✅ Sync UI indicators (showing placeholder state)
- ✅ Auto-save with debouncing
- ✅ Background sync queue (marks for sync but doesn't upload yet)

## What Requires AWS SDK

- ⏳ Upload drawings to S3
- ⏳ Download drawings from S3
- ⏳ Multi-device sync
- ⏳ Conflict resolution with remote drawings

---

## Step 1: Add AWS SDK for Swift

### Option A: Via Xcode UI (Recommended)

1. Open `PDFMarkup.xcodeproj` in Xcode
2. Select the project in the navigator
3. Select the "PDFMarkup" target
4. Go to the "Package Dependencies" tab
5. Click the "+" button
6. Enter package URL: `https://github.com/awslabs/aws-sdk-swift`
7. Click "Add Package"
8. Select the following products:
   - `AWSS3`
   - `AWSClientRuntime`
   - `ClientRuntime`
9. Click "Add Package"

### Option B: Manual Package.swift (If Using SPM)

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/awslabs/aws-sdk-swift", from: "0.40.0"),
],
targets: [
    .target(
        name: "PDFMarkup",
        dependencies: [
            .product(name: "AWSS3", package: "aws-sdk-swift"),
        ]
    ),
]
```

---

## Step 2: Configure AWS Credentials

You have three options for providing AWS credentials:

### Option A: Environment Variables (Development)

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_REGION="us-east-1"  # or your region
```

Then launch the app from terminal:
```bash
open PDFMarkup.app
```

### Option B: AWS Credentials File (Recommended for Local Development)

Create `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = your-access-key-id
aws_secret_access_key = your-secret-access-key
```

Create `~/.aws/config`:

```ini
[default]
region = us-east-1
```

### Option C: Hardcode in App (NOT RECOMMENDED - Security Risk)

Only for testing, never for production:

```swift
// In DrawingSyncManager.swift
let config = try await S3Client.S3ClientConfiguration(
    credentialsProvider: StaticCredentialsProvider(
        accessKeyId: "YOUR_KEY",
        secretAccessKey: "YOUR_SECRET"
    ),
    region: "us-east-1"
)
```

---

## Step 3: Uncomment AWS SDK Code

In `PDFMarkup/DrawingSyncManager.swift`, uncomment the following sections:

### 1. Import statements (lines 11-14):
```swift
import AWSS3
import AWSClientRuntime
import ClientRuntime
```

### 2. Upload function (lines 51-70):
```swift
// Uncomment the full uploadDrawings implementation
```

### 3. Download function (lines 96-122):
```swift
// Uncomment the full downloadDrawings implementation
```

---

## Step 4: Configure S3 Bucket Permissions

Your S3 bucket needs the following permissions for the IAM user:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::jimmyhmiller-bucket/drawings/*",
                "arn:aws:s3:::jimmyhmiller-bucket"
            ]
        }
    ]
}
```

### Creating an IAM User

1. Go to AWS IAM Console
2. Create a new user: `pdfmarkup-sync-user`
3. Attach the policy above (inline or managed)
4. Create access keys
5. Save the access key ID and secret access key

---

## Step 5: Rebuild and Test

1. Clean build folder: `Product > Clean Build Folder` (⇧⌘K)
2. Rebuild: `Product > Build` (⌘B)
3. Run the app
4. Open a PDF and draw some markups
5. Click the sync button (circular arrows icon)
6. Check console for sync logs:
   ```
   Successfully uploaded drawings for {hash} to S3
   Successfully downloaded drawings for {hash} from S3
   ```

---

## Verification Checklist

### Test Local Sync
- [ ] Draw markups on a PDF
- [ ] Wait 2 seconds (auto-save)
- [ ] Quit and relaunch app
- [ ] Markups still visible ✅

### Test S3 Upload (After AWS SDK Setup)
- [ ] Draw markups on a PDF
- [ ] Click sync button
- [ ] Check AWS S3 Console for `drawings/{hash}.drawings.json`
- [ ] File should contain JSON with drawing data

### Test S3 Download (Multi-Device)
- [ ] Upload drawings from Device A
- [ ] Open same PDF on Device B
- [ ] Click sync button
- [ ] Drawings from Device A appear on Device B

### Test Conflict Resolution
- [ ] Draw different markups on same PDF on two devices
- [ ] Sync both
- [ ] The newer version should win (last-write-wins)

---

## S3 Storage Structure

After setup, your S3 bucket will have:

```
s3://jimmyhmiller-bucket/
├── pdfs/
│   ├── ai/
│   ├── books/
│   └── computer-philosophy/
│       └── {hash}.pdf
└── drawings/               # New directory
    ├── {hash1}.drawings.json
    ├── {hash2}.drawings.json
    └── {hash3}.drawings.json
```

Each `.drawings.json` file contains:
```json
{
  "pdfHash": "abc123...",
  "lastModified": "2025-12-01T22:00:00Z",
  "pages": {
    "0": "base64-encoded-PKDrawing-data",
    "1": "base64-encoded-PKDrawing-data"
  }
}
```

---

## Troubleshooting

### Build Errors

**Error**: `Cannot find 'AWSS3' in scope`
- **Solution**: Make sure you added the AWS SDK package dependency

**Error**: `No such module 'AWSS3'`
- **Solution**: Clean build folder and rebuild

### Runtime Errors

**Error**: `Failed to upload drawings: NoCredentialsError`
- **Solution**: Check that AWS credentials are configured (Step 2)

**Error**: `Failed to upload drawings: AccessDenied`
- **Solution**: Check IAM permissions (Step 4)

**Error**: `Failed to upload drawings: NoSuchBucket`
- **Solution**: Verify bucket name is correct in `DrawingSyncManager.swift`

### Sync Not Working

1. **Check console logs** for error messages
2. **Verify credentials** are set correctly
3. **Check network** - sync requires internet connection
4. **Verify bucket permissions** in IAM
5. **Check bucket region** matches configuration

---

## Cost Estimation

AWS S3 pricing for this app (approximate):

- **Storage**: $0.023 per GB/month
  - Each drawing file: ~5-50 KB
  - 1000 PDFs with drawings: ~25 MB = $0.0006/month

- **Requests**:
  - PUT (upload): $0.005 per 1000 requests
  - GET (download): $0.0004 per 1000 requests
  - 100 syncs/day = 3000/month = $0.015 + $0.001 = $0.016/month

**Total estimated cost**: < $0.02/month for typical usage

---

## Next Steps

After AWS SDK is set up, you can:

1. **Enable automatic sync**: Currently syncs every 30 seconds for changed PDFs
2. **Monitor sync status**: Cloud icon in toolbar shows sync state
3. **Manual sync**: Click the circular arrows button
4. **View logs**: Check console for detailed sync information

## Questions?

If you encounter issues:
1. Check the troubleshooting section above
2. Review AWS CloudWatch logs for S3 access
3. Verify IAM permissions match the required policy
4. Check that the bucket region matches your configuration
