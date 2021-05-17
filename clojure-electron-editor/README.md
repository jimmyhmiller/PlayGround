# Clojure Electron Editor

## How to Run
```
npm install electron -g
npm install shadow-cljs -g
npm install

npm run dev
electron .
```

## Release
```
npm run build
electron-packager . HelloWorld --platform=darwin --arch=x64
```
