import { call, put, takeEvery, takeLatest, select, debounce } from 'redux-saga/effects'
import { groupBy, property, mapValues as mapValuesPrime, compose, values, flatMap } from 'lodash/fp';
import { stripMargin } from 'stripmargin';

import { UPDATE_CODE, createComponent, deleteComponent, updateComponentMetadata, exportCode } from './actions';

const mapValues = mapValuesPrime.convert({ 'cap': false });


const extractComponents = (code) => {
  const compRegex = /<([A-Z][a-zA-Z0-9]+).*\/?>?/g
  const propsRegex = /([A-Za-z]+)=/g

  // This is a complete mess,  but is shows me the sorts of issues
  // I will have to deal with to make these derived properties generic.
  // I also need to no delete things if the props are referenced in the code.
  // and I need to be able to get rid of unused props.

  // I guess I need to consider different kinds of derivation. 
  // How to derive properties? How do they combine? 
  // What do we do when outside code mentions something?
  // What about when code in an editor mentions something that doesn't exist?
  // How do we show used vs unused?


  // Also, I am really tempted to pull in fluent compose. It would be make this beautiful.
  return compose(
    values,
    mapValues((values, name) => ({
      name,
      props: [...new Set(flatMap(property("props"), values))]
    })),
    groupBy(property("name"))
  )([...code.matchAll(compRegex)]
      .map(x => ({
        name: x[1],
        props: [...x[0].matchAll(propsRegex)].map(x => x[1])
      })))
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

  // Needs refactoring to be generic.
  // What are the key features I care about?

  const allComponents = yield select(state => Object.values(state.editors).filter(c => c.type = "component"))
  const allCode = extractAllCode(allComponents);

  const unchangedComponents = allComponents.filter(c => (c.name === c.code || c.code === "") && c.name !== "Main" );

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
    } else {
      yield put(updateComponentMetadata({ name, props }))
    }
  }
}

const deriveCode = function* () {
  const editors = yield select(state => Object.values(state.editors));
  const componentCode = editors
    .filter(e => e.type === "component")
    .map(({ code, name, props }) => stripMargin(`
      |  const ${name} = (${props && props.length > 0  ? "{ " + props.join(", ") + " }" : "props"}) => {
      |    return <>${code}</>
      |  }`)
    )
    .join("\n\n");


  yield put(exportCode({ code: componentCode }))

}

const editorSaga = function* () {
  yield debounce(100, UPDATE_CODE, deriveComponents);
  yield debounce(100, UPDATE_CODE, deriveCode);
}

export default editorSaga