import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as nodejs from 'aws-cdk-lib/aws-lambda-nodejs';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as path from 'path';

export class BookshelfCloudStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Both credentials supplied via CDK context at deploy time:
    //   cdk deploy -c jwtSecret=$(openssl rand -hex 32) \
    //              -c username=admin \
    //              -c bcryptHash=<bcrypt hash of the password>
    // The values are written into Secrets Manager and the Lambda reads them
    // at runtime. No more placeholder-substitution in source.
    const jwtSecret = this.node.tryGetContext('jwtSecret') as string | undefined;
    const username = this.node.tryGetContext('username') as string | undefined;
    const bcryptHash = this.node.tryGetContext('bcryptHash') as string | undefined;
    if (!jwtSecret || jwtSecret.length < 32 || !username || !bcryptHash) {
      throw new Error(
        'Missing context. Pass: -c jwtSecret=<hex32+> -c username=<name> -c bcryptHash=<hash>',
      );
    }

    const bucket = new s3.Bucket(this, 'LibraryBucket', {
      bucketName: `bookshelf-cloud-${this.account}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: 'ratelimit-gc',
          prefix: 'ratelimit/',
          expiration: cdk.Duration.days(1),
        },
      ],
    });

    const credentials = new secretsmanager.Secret(this, 'Credentials', {
      secretName: 'bookshelf-cloud/credentials',
      description: 'JWT HMAC secret + bcrypt-hashed user password',
      secretObjectValue: {
        jwtSecret: cdk.SecretValue.unsafePlainText(jwtSecret),
        username: cdk.SecretValue.unsafePlainText(username),
        bcryptHash: cdk.SecretValue.unsafePlainText(bcryptHash),
      },
    });

    const fn = new nodejs.NodejsFunction(this, 'HandlerFn', {
      entry: path.join(__dirname, '..', 'functions', 'handler', 'index.mjs'),
      handler: 'handler',
      runtime: lambda.Runtime.NODEJS_22_X,
      // 1024 MB → full vCPU. Needed so bcrypt cost-12 completes in well
      // under iOS's auth-challenge timeout (~2s). At 256 MB the same hash
      // took 2-3s and BookPlayer treated the slow response as -1013
      // (NSURLErrorUserAuthenticationRequired).
      memorySize: 1024,
      timeout: cdk.Duration.seconds(15),
      environment: {
        CREDENTIALS_SECRET_ARN: credentials.secretArn,
        // ~100 years. Effectively "never expire" — BookPlayer has no
        // refresh flow so any TTL means periodic forced re-login. If the
        // device is lost, rotate the JWT secret to invalidate all tokens.
        JWT_TTL_SECONDS: String(60 * 60 * 24 * 365 * 100),
        BUCKET_NAME: bucket.bucketName,
      },
      bundling: {
        format: nodejs.OutputFormat.ESM,
        target: 'node22',
        nodeModules: ['bcryptjs'],
      },
    });
    credentials.grantRead(fn);
    // Read access to manifests + signing of presigned URLs for media.
    bucket.grantRead(fn);
    // Write access only to the rate-limit prefix.
    bucket.grantWrite(fn, 'ratelimit/*');

    const url = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
    });

    new cdk.CfnOutput(this, 'BucketName', { value: bucket.bucketName });
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: url.url,
      description: 'Paste this (without trailing slash) into BookPlayer as the server URL',
    });
    new cdk.CfnOutput(this, 'CredentialsSecretArn', { value: credentials.secretArn });
  }
}
