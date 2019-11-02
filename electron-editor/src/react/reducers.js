import { setIn, updateIn, unset, update } from 'zaphod/compat';
import prettier from "prettier/standalone";
import parserBabel from "prettier/parser-babylon";

import {
  UPDATE_CODE,
  PRETTIFY_CODE,
  CREATE_COMPONENT,
  DELETE_COMPONENT,
  UPDATE_COMPONENT_METADATA,
  EXPORT_CODE,
} from "./actions";


const tryPrettier = (code) => {
  try {
    return prettier.format(code, { parser: "babel", plugins: [parserBabel] })
  }
  catch (e) {
    return code;
  }
}

const updateCode = (state, { code, name }) => setIn(state, [name, "code"], code)

const prettify = (state, { name }) =>
  updateIn(
    state,
    [name, "code"],
    tryPrettier,
  );


const createComponent = (state, { code, name, props }) =>
  updateIn(
    state,
    [name],
    component => component || {  type: "component", code, name, props }
  );

const deleteComponent = (state, { code, name }) =>
  unset(state, name)

const updateComponentMetadata = (state, { name, props }) =>
  setIn(state, [name, "props"], props)



const initialEditorState = {
  Main: {
    name: "Main",
    color: "#2a2f38",
    code: "",
    type: "component"
  }
}


export const editorReducer = (state=initialEditorState, action) => {
  switch (action.type) {
    case UPDATE_CODE: {
     return updateCode(state, action)
    }
    case PRETTIFY_CODE: {
      return prettify(state, action)
    }
    case CREATE_COMPONENT: {
      return createComponent(state, action)
    }
    case DELETE_COMPONENT: {
      return deleteComponent(state, action)
    }
    case UPDATE_COMPONENT_METADATA: {
      return updateComponentMetadata(state,action)
    }
    default: {
      return state;
    }
  }
}

const exportCode = (state, { code }) => ({
  ...state,
  code,
})

export const exportReducer = (state="", action) => {
  switch (action.type) {
    case EXPORT_CODE: {
      return exportCode(state, action)
    }
    default: {
      return state;
    }
  }
}



