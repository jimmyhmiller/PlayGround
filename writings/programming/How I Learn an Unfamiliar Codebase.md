# How I Learn an Unfamiliar Codebase

The biggest shock of my early career was just how much code I needed to read that other wrote. I had never dealt with this. I had a hard enough time understanding my own code sometimes. The idea of understanding hundreds of thousands or even millions of lines of code written by countless other people scared me. What I quickly learned is that you don't have to understand a codebase in its entirety to be effective in it. But just saying that is not super helpful. So rather than tell, I want to show.

In this post I'm going to walk you through how I learn an unfamiliar codebase. I've chosen one whose general purpose and scope I'm generally familiar with. But one that I've never contributed to or read, [Next.js](https://github.com/vercel/next.js). But I've chosen to be a bit more particular than that. I'm particularly interested in learn more about the rust bundler setup (turbopack) that nextjs has been building out. So that's where we will concentrate our time.

## Being Clear About our Goal

Trying to learn a codebase is distinctly different from trying to simply fix a bug or add a feature. We may use bugs, talk about features, make changes, etc. But we are not trying to contribute to the codebase, yet. Instead, we are trying to get our mind around how the codebase generally works. We aren't concerning ourselves with things like coding standards, common practices, or the development roadmap. We aren't even concerned with correctness. The changes we make are about seeing how the codebase responds so we can make sense of it.

### Starting Point

I find starting at `main` to be almost always completely unhelpful. At least for how my brain works about a codebase. From main, yes we have a single entrypoint. But now we are asking ourselves to understand the whole. But it is worse than that when dealing with a large codebase. What even is the real main we are concerned with? Let's start with our target library and figure out what it even consists of. So let's start by listing out our directories

```bash
❯ ls -1d */
apps/
bench/
contributing/
crates/
docs/
errors/
examples/
packages/
patches/
rspack/
scripts/
test-config-errors/
test/
turbo/
turbopack/
```

A couple things to note. We have packages, crates, turbo, and turbopack. Crates are relevant here because we know we are interested in some of the rust code, but we are also interested in turbopack in particular. A quick look at these shows that turbo, packages, and crates are probably not our target. Why do I say that? Because turbopack has its own crates folder.

```bash
❯ ls -d turbopack/crates/* | wc -l
      54
```

So there are 54 crates under turbopack.... This is beginning to feel a bit daunting. So why don't we take a step back and find a better starting point. One starting point I find particularly useful is a [bug report](https://github.com/vercel/next.js/issues/88009).

![Screenshot 2026-01-11 at 10.26.15 AM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2026-01-11 at 10.26.15 AM.png)

I found this by simply looking at at recently open issues. When I first found it, it had no comments on it. In fact, I find bug reports with only reproducing instructions to be the most useful. Remember, we are trying to learn, not fix a bug.

So I spent a little time looking at the bug report, it is fairly clear. It does indeed reproduce. But it has a lot of code. So as is often the case, it is useful to reproduce it to the minimal case. So that's what I did. Here is the important code and the problem we are using to learn from.

```tsx
// app.ts
import { greeting } from './utility'

export default function Home() {
  return (
    <div>${greeting} World</div>
  );
}
```

```tsx
// utility.ts
export enum MyEnum {
  UNIQUE_1 = 'UNIQUE_1',
  UNIQUE_2 = 'UNIQUE_2',
}

export const greeting = "Hello";
```

MyEnum here is dead code. It should not show up in our final bundle. But when we do `next build` and look for it we get:

```bash
❯ rg --no-ignore --hidden -g '!*.map' UNIQUE_1
app/utility.ts
2:  UNIQUE_1 = 'UNIQUE_1',

.next/server/chunks/ssr/[root-of-the-server]__5f0b8324._.js
1:module.exports=[93695,(a,b,c)=>{b.exports=a.x("next/dist/shared/lib/no-fallback-error.external.js",()=>require("next/dist/shared/lib/no-fallback-error.external.js"))},70864,a=>{a.n(a.i(33290))},43619,a=>{a.n(a.i(79962))},13718,a=>{a.n(a.i(85523))},18198,a=>{a.n(a.i(45518))},50708,a=>{"use strict";var b,c=a.i(7997);function d(){return(0,c.jsxs)("div",{children:["$","Hello"," World"]})}(b={}).UNIQUE_1="UNIQUE_1",b.UNIQUE_2="UNIQUE_2",a.s(["default",()=>d],50708)}];
```

If we instead do `next build --webpack`

```bash
❯ rg --no-ignore --hidden -g '!*.map' UNIQUE_1
app/utility.ts
2:  UNIQUE_1 = 'UNIQUE_1',
```

The code is completely gone from our build. 

### Setting an Agenda

So now we have our bug. But remember. Our goal here is not to fix the bug. But to understand the code. So our goal is going to be to use this little mini problem to understand what code is involved in this bug. To understand the different ways we could fix this bug. To understand why this bug happened in the first place. To understand some small slice of the turbopack codebase.

So at each junction, we are going to resist the urge to simply find the offending code. We are going to take detours. We are going to ask questions. Our hope is from the start of this process to the end we no longer think of the code involved in this action as a blackbox. But we will, intentionally leave ourselves with open questions. As I write these words, I have no idea where this will take us. I have not prepared this ahead of time. I am not telling you a fake tale from a codebase I already know. Yes, I will simplify and skip parts. But you will come along with me.

## Finding Tree Shaking

The first step for understanding any project is getting some part of it running. Well, I say that, but in my day job I've been at companies where this is a multi day, or week long effort. Sometimes because of lack of access, sometimes from unclear instructions, if you find yourself in that situation, you now have a new task, understand it well enough to get it to build. Well, unfortunately, that is the scenario we find ourselves in.

### The First Side Quest

I can't think of a single one of these endeavors I've gone on to learn a codebase that didn't involve a completely undesireable, momentum stopping side quest. For this one it was as soon as I tried to make changes to the turbopack rust code and get it working in my test project. There are [instructions on how to do this](https://github.com/vercel/next.js/blob/25046dfa1bd78b1f8da478ceda51ae86c07b673c/contributing/core/developing.md#testing-a-local-nextjs-version-on-an-application). In fact, we even get an explanation on why it is a bit weird.

> Since Turbopack doesn't support symlinks when pointing  outside of the workspace directory, it can be difficult to develop  against a local Next.js version. Neither `pnpm link` nor `file:` imports quite cut it. An alternative is to pack the Next.js version you want to test into a tarball and add it to the pnpm overrides of your  test application. The following script will do it for you:
>
> ```
> pnpm pack-next --tar && pnpm unpack-next path/to/project
> ```

Okay, straight forward enough. I start by finding somewhere in the turbopack repo that I *think* will be called more than once and I add the following:
```diff
diff --git a/turbopack/crates/turbopack/src/lib.rs b/turbopack/crates/turbopack/src/lib.rs
index 20891acf86..8691f1951c 100644
--- a/turbopack/crates/turbopack/src/lib.rs
+++ b/turbopack/crates/turbopack/src/lib.rs
@@ -80,6 +80,7 @@ async fn apply_module_type(
     css_import_context: Option<ResolvedVc<ImportContext>>,
     runtime_code: bool,
 ) -> Result<Vc<ProcessResult>> {
+    println!("HERERE!!!!!!!");
     let tree_shaking_mode = module_asset_context
         .module_options_context()
         .await?
```

Yes. very scientific I know. But I've found this to be a rather effective method of making sure my changes are picked up. So I do that, make sure I've built and done necessary things. I run

```bash
pnpm swc-build-native && pnpm pack-next --tar && pnpm unpack-next /Users/jimmyhmiller/Documents/Code/open-source/turbo-treeshake2
```

Then that script tells me to

```bash
Add the following overrides to your workspace package.json:
  "pnpm": {
    "overrides": {
      "next": "file:/Users/jimmyhmiller/Documents/Code/open-source/next.js-88009/tarballs/next.tar",
      "@next/mdx": "file:/Users/jimmyhmiller/Documents/Code/open-source/next.js-88009/tarballs/next-mdx.tar",
      "@next/env": "file:/Users/jimmyhmiller/Documents/Code/open-source/next.js-88009/tarballs/next-env.tar",
      "@next/bundle-analyzer": "file:/Users/jimmyhmiller/Documents/Code/open-source/next.js-88009/tarballs/next-bundle-analyzer.tar"
    }
  }

Add the following dependencies to your workspace package.json:
  "dependencies": {
    "@next/swc": "file:/Users/jimmyhmiller/Documents/Code/open-source/next.js-88009/tarballs/next-swc.tar",
    ...
  }


```

I go to build my project and **HERERE!!!!!!!** does not show up at all...

### Understanding the Build System

I will save you the fun details here of looking through this system. But I think its important to mention a few things. First, `@next/swc` being a dependency immeditately stoodout to me. In my day job I maintain a fork of swc (WHY???) for some custom stuff. I definitely won't pretend to be an expert on swc, but I know its written in rust. I know it's a native dependency. The changes I'm making are native dependencies. But I see no mention at all of turbopack. In fact, if I search in my test project I get the following:

```bash
~/Documents/Code/open-source/turbo-treeshake2 main⚡
❯ fd --no-ignore swc      
node_modules/@next/swc

~/Documents/Code/open-source/turbo-treeshake2 main⚡
❯ fd --no-ignore turbopack
# empty
```

So I have a sneaking suspicion my turbopack code should be in that tar. So let's look at the tar

```bash
❯ du -h /Users/jimmyhmiller/Documents/Code/open-source/next.js-88009/tarballs/next-swc.tar
 12k
```

Ummm. That seems a bit small... Let's look what's inside.

```bash
❯ tar -tf /Users/jimmyhmiller/Documents/Code/open-source/next.js-88009/tarballs/next-swc.tar
./README.md
./package.json
```

Okay, I think we found our problem.

#### Regex - Now You Have Two Problems

After lots of searching the culprit came down to:

```typescript
const packageFiles = [...globbedFiles, ...simpleFiles].sort()
const set = new Set()
return packageFiles.filter((f) => {
  if (set.has(f)) return false
  // We add the full path, but check for parent directories too.
  // This catches the case where the whole directory is added and then a single file from the directory.
  // The sorting before ensures that the directory comes before the files inside of the directory.
  set.add(f)
  while (f.includes('/')) {
    f = f.replace(/\/[^/]+$/, '')
    if (set.has(f)) return false
  }
  return true
})
```

In our case the input came from [this file](https://github.com/vercel/next.js/blob/25046dfa1bd78b1f8da478ceda51ae86c07b673c/packages/next-swc/package.json#L1C1-L7C5) and f was `"native/"`. Unfortunately, this little set + regex setup causes `native/` to be filtered out. Why? Because it doesn't match the regex. This regex is look for a `/` with characters after it. We have none. So since we are already in the set (we just added ourselves) we filter ourselves out.

How do we solve this problem? There are countless answers really. I had claude whip me up one without regex.

```typescript
const packageFiles = [...globbedFiles, ...simpleFiles].sort()
const set = new Set<string>()
return packageFiles.filter((f) => {
  const normalized = join(f)
  if (set.has(normalized)) return false
  // We add the normalized path, but check for parent directories too.
  // This catches the case where the whole directory is added and then a single file from the directory.
  // The sorting before ensures that the directory comes before the files inside of the directory.
  set.add(normalized)
  let parent = dirname(normalized)
  while (parent !== '.' && parent !== '/') {
    if (set.has(parent)) return false
    parent = dirname(parent)
  }
  return true
})
```

But my gut says the sorting let's us do this much simpler. If the directory comes first in the sorting order, we should be able to do something like check for prefix, but does that requires us to make sure people have a trailing `/` if it is a directory so we don't include `turbopack/` because someone wanted to include `turbo`? I  left that aside. I plan on filing a bug about it. But after this change we can finally see **HERERE!!!!!!!** a lot.

## For Real This Time?

Okay, we now have something we can test. But where do we even begin? This is one reason we choose this bug. It gives a few avenues to go down. First the report says that these enums are not being "tree-shaken" is that the right term? One thing I've learned from experience is to never assue that the end user is using terms in the same manner as the codebase. So this can be a starting point, but it might be wrong. 

With some searching around we can actually see that there is a configuration for turning turbopackTreeShaking on or off

```typescript
const nextConfig: NextConfig = {
  experimental: {
    turbopackTreeShaking: true,
  },
};
```

It was actually a bit hard out exactly where the default for this was. It isn't actually documented. So let's just enable it and see what we get.

```bash
> Build error occurred
Error [TurbopackInternalError]: Failed to write app endpoint /page

Caused by:
- index out of bounds: the len is 79 but the index is 79

Debug info:
- Execution of get_all_written_entrypoints_with_issues_operation failed
- Execution of EntrypointsOperation::new failed
- Execution of all_entrypoints_write_to_disk_operation failed
- Execution of Project::emit_all_output_assets failed
- Execution of *emit_assets failed
- Execution of all_assets_from_entries_operation failed
- Execution of *all_assets_from_entries failed
- Execution of output_assets_operation failed
- Execution of <AppEndpoint as Endpoint>::output failed
- Failed to write app endpoint /page
- Execution of AppEndpoint::output failed
- Execution of whole_app_module_graph_operation failed
- Execution of *ModuleGraph::from_single_graph failed
- Execution of *SingleModuleGraph::new_with_entries failed
- Execution of Project::get_all_entries failed
- Execution of <AppEndpoint as Endpoint>::entries failed
- Execution of get_app_page_entry failed
- Execution of *ProcessResult::module failed
- Execution of <ModuleAssetContext as AssetContext>::process failed
- Execution of EcmascriptModulePartAsset::select_part failed
- Execution of split_module failed
- index out of bounds: the len is 79 but the index is 79
    at <unknown> (TurbopackInternalError: Failed to write app endpoint /page) {
  type: 'TurbopackInternalError',
  location: 'turbopack/crates/turbopack-ecmascript/src/tree_shake/graph.rs:745:16'
}
```

Well, I think we figured out that the default is off. So one option is that we never "tree shake" anything. But that seems wrong. At this point I looked into tree shaking a bit in the codebase and while I started to understand a few things, I've been at this point before. Sometimes it is good to go deep. But how much of this codebase do I really understand? If tree shaking is our culprit (seeming unlikely at this point), it might be good to know how code gets there.

## How a Chunk is Made

Our "search around the codebase" strategy failed. So now we try a different tactic. We know a couple things.

1. Our utilities.ts file is read and parsed
2. It ends up in a file under a "chunks" directory.

We now have two points we can use try to trace what happens. Let's start with parsing. Luckily here it is straightforward `parse_file_content`. When we look at this code, we can see that swc does the heavy lifting. First it parses it into a typescript ast, then applies transforms to turn it into javascript. At this point point we don't write to a string, but if you edit the code and use an emitter you see this:

```javascript
export var MyEnum = /*#__PURE__*/ function(MyEnum) {
    MyEnum["UNIQUE_1"] = "UNIQUE_1";
    MyEnum["UNIQUE_2"] = "UNIQUE_2";
    return MyEnum;
}({});
export const greeting = "Hello!";
```

Now to find where we write the chunks. In most programs this would be pretty easy. Typically there is a linear flow somewhere that just shows you the steps. Or if you can't piece one together, you can simply breakpoint and follow the flow. But turbopack as a rather advanced system involving async rust (more on this later). So in keeping with the tradition of not trying to do things the rely too heavily on my knowledge, I have did the tried and true, log random things until they look relavant. And what I found made realize that logging was not going to be enough. It was time to do my tried and true learning technique, visualization

## Building a Visualizer

Ever since my [first job](https://jimmyhmiller.com/ugliest-beautiful-codebase) I have been building custom tools to visualize codebases. Perhaps this is due to my aphantasia. I'm not really sure. Some of these visualizers make there way into general use for me. But more often than not they are a means of understanding. When I applied for a job at Shopify working on YJIT I built a [simple visualizer](https://jimmyhmiller.com/yjit) but never got around to make it more useful than a learning tool. The same thing is true here but this time thanks to AI it looks a bit more professional.

This time we want to give a bit more structure to what we'd do with print. We are trying to get events out that have a bunch of information. Mostly we are interested in files and their contents over time. Looking through the codebase we find that one key abstract is an ident, this will help us indentiy files. We will simple find points that seem interesting, make a corresponding event, make sure it has idents associated with it and send that event over websocket. 

Then with that raw information we can have our visualizer stitch together the what exactly happens. [add foot note about the various things next has].





