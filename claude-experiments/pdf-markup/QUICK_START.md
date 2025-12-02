# S3 Sync - Quick Start (2 Steps!)

## Your PDF Markup app is ready! Just add AWS credentials to enable cloud sync.

### ✅ What Already Works (No Setup Needed):
- Local drawings that persist across app restarts
- Export PDFs with markups baked in
- Sync UI with cloud icons and status
- Auto-save with background sync queue

### ⏰ What Needs AWS Credentials (2-Minute Setup):
- Actual upload to S3
- Download from S3
- Multi-device sync

---

## Step 1: Get AWS Credentials (5 minutes)

1. Go to **AWS IAM Console**: https://console.aws.amazon.com/iam/
2. Click "Users" → "Create user"
3. Username: `pdfmarkup-sync`
4. Click "Next"
5. Select "Attach policies directly"
6. Click "Create policy" → "JSON" tab
7. Paste this policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::jimmyhmiller-bucket/drawings/*"
        }
    ]
}
```

8. Name it: `PDFMarkupS3Access`
9. Create policy
10. Go back and attach this policy to your user
11. Finish creating user
12. Click on the user → "Security credentials"
13. "Create access key" → "Application running outside AWS" → Next
14. **Copy the Access Key ID and Secret Access Key**

---

## Step 2: Paste Credentials (1 minute)

1. Open `PDFMarkup/AWSCredentials.swift`
2. Replace the placeholder values:

```swift
static let accessKeyId = "PASTE_YOUR_ACCESS_KEY_ID_HERE"
static let secretAccessKey = "PASTE_YOUR_SECRET_ACCESS_KEY_HERE"
```

3. Save the file
4. Build and run (⌘R)

**That's it!** ✨

---

## Testing S3 Sync

1. Draw markups on a PDF
2. Click the sync button (circular arrows icon)
3. Check console for: `✅ Successfully uploaded drawings...`
4. Check S3 Console: You should see `drawings/{hash}.drawings.json`

---

## Troubleshooting

**"AWS credentials not configured"**
→ Check that you saved AWSCredentials.swift with real values (not "YOUR_...")

**"Failed to upload to S3: Access Denied"**
→ Check IAM policy has `s3:PutObject` permission for `drawings/*`

**"Failed to upload to S3: No Such Bucket"**
→ Change `bucket` in AWSCredentials.swift to match your bucket name

**Want different region?**
→ Change `region` in AWSCredentials.swift (e.g., "us-west-2")

---

## Cost

S3 pricing for this app:
- **Storage**: $0.023/GB/month → ~$0.0006/month for 1000 PDFs
- **Requests**: $0.005/1000 PUTs → ~$0.015/month for typical usage
- **Total**: < $0.02/month 💰

---

## Security Note

⚠️ **Add to `.gitignore`:**
```
PDFMarkup/AWSCredentials.swift
```

This keeps your credentials private if you commit to Git!

---

## That's It!

No AWS SDK. No complex setup. Just paste credentials and go! 🚀
