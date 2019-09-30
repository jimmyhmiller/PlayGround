import { call, put, takeEvery, takeLatest, select, debounce } from 'redux-saga/effects'

import { UPDATE_CODE, createComponent, deleteComponent } from './actions';


const extractComponents = (code) => {
  const compRegex = /<([A-Z][a-zA-Z0-9]+).*\/?>?/g
  const propsRegex = /([A-Za-z]+)=/g
  return [...code.matchAll(compRegex)]
    .map(x => ({
      name: x[1],
      props: [...x[0].matchAll(propsRegex)].map(x => x[1])
    }))
 }


const extractAllCode = (components) => {
  return components
      .map(c => c.code)
      .join("\n");
}

const setDifference = (set1, set2) => {
  return new Set([...set2].filter(x => !set1.has(x)))
}


const deriveComponents = function* ({ code, name }) {

  const allComponents = yield select(state => Object.values(state.editors).filter(c => c.type = "component"))
  const allCode = extractAllCode(allComponents);

  const unchangedComponents = allComponents.filter(c => c.name === c.code);

  const componentsInCode = extractComponents(allCode);

  const notInCodeAndUnchanged = setDifference(
    new Set(componentsInCode.map(x => x.name)),
    new Set(unchangedComponents.map(x => x.name)),
  )

  for (let name of notInCodeAndUnchanged) {
    yield put(deleteComponent({ name }))
  }

  for (let { name, props } of componentsInCode) {
    const exists = yield select(state => !!state.editors[name])
    if (!exists) {
      yield put(createComponent({ code: name, name, props }))
    }
  }

}

const editorSaga = function* () {
  yield debounce(100, UPDATE_CODE, deriveComponents);
}

export default editorSaga