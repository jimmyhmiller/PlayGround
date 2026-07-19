# Native dark-reference audit

The Flowline catalog was compared against the upstream Native SDK dark
references in `docs/public/components`. The audit used the base dark reference
for every documented component, with the additional icon references checked
against the generated 51-icon atlas.

The examples intentionally use Flowline workflow data instead of copying the
documentation examples. Fidelity is evaluated at the component boundary:
padding, corner treatment, border and fill tokens, typography hierarchy,
grouping, focus and selected states, overlay planes, and component-specific
chrome.

| Group | Components checked |
| --- | --- |
| Controls | Button, Checkbox, Input, Radio group, Slider, Switch, Textarea, Toggle |
| Surfaces | Alert, Bubble, Card, Dialog, Drawer, Panel, Popover, Sheet, Tooltip |
| Navigation | Accordion, Breadcrumb, Button group, Combobox, Dropdown menu, Pagination, Select, Tabs |
| Content | Avatar, Badge, Chart, Icon, Image, Markdown, Progress, Separator, Skeleton, Spinner, Status bar, Timeline |
| Collections and layout | Input group, List, Resizable, Scroll, Spacer, Split, Stepper, Table, Tree, Virtual list |

Corrections made from the comparison:

- Table grid boxes were removed in favor of Native row hairlines and selected-row washes.
- Bubble width now hugs its message, and reactions dock across the lower edge without consuming layout space.
- Breadcrumb slash separators were replaced with muted chevrons and ancestor press targets.
- Input Group now uses the composer composition: multiline field, attachment affordance, and trailing send control.
- Tooltip display now observes the upstream hover delay instead of appearing immediately.
- Tree rows use a visible roving-focus wash and hierarchy-aware Left/Right behavior.
- Virtual-list labels are bound to the current calculated window instead of static example strings.
- The complete upstream icon inventory replaces the three-symbol placeholder.
- Markdown includes native link dispatch and model-owned details expansion.

The reproducible catalog captures are `native-controls.png`,
`native-surfaces.png`, `native-navigation.png`, `native-overlays.png`,
`native-content.png`, `native-collections.png`, and
`native-primitives.png`.
