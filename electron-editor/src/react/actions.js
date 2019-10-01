export const UPDATE_CODE = 'UPDATE_CODE';

export const updateCode = ({ code, name }) => ({
  type: UPDATE_CODE,
  code,
  name,
})

export const PRETTIFY_CODE = 'PRETTIFY_CODE';

export const prettifyCode = ({ name }) => ({
  type: PRETTIFY_CODE,
  name,
})


export const CREATE_COMPONENT = 'CREATE_COMPONENT';

export const createComponent = ({ name, code, props }) => ({
  type: CREATE_COMPONENT,
  name,
  code,
  props,
})


export const DELETE_COMPONENT = 'DELETE_COMPONENT';

export const deleteComponent = ({ name }) => ({
  type: DELETE_COMPONENT,
  name,
})

export const UPDATE_COMPONENT_METADATA = 'UPDATE_COMPONENT_METADATA';

export const updateComponentMetadata = ({ name, props }) => ({
  type: UPDATE_COMPONENT_METADATA,
  name,
  props,
})

