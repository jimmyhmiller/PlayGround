#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { BookshelfCloudStack } from '../lib/bookshelf-cloud-stack';

const app = new cdk.App();
new BookshelfCloudStack(app, 'BookshelfCloudStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  description: 'Static-hosted AudiobookShelf mock for BookPlayer iOS',
});
