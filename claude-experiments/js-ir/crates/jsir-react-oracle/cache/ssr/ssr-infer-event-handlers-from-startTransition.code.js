// @outputMode:"ssr"
function Component() {
  useTransition();
  const state = 0;
  const ref = useRef(null);
  const onChange = undefined;

  return <CustomInput value={state} onChange={onChange} ref={ref} />;
}
