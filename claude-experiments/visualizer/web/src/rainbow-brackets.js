import { EditorView, Decoration, ViewPlugin } from '@codemirror/view';

const colors = [
    '#c0caf5', // level 0 — soft white (blends with base text)
    '#7aa2f7', // level 1 — blue
    '#bb9af7', // level 2 — purple
    '#7dcfff', // level 3 — cyan
    '#e0af68', // level 4 — gold
    '#9ece6a', // level 5 — green
    '#f7768e', // level 6 — pink (not error-red)
];

const classNames = colors.map((_, i) => `rb-depth-${i}`);

const plugin = ViewPlugin.fromClass(class {
    decorations;

    constructor(view) {
        this.decorations = this.build(view);
    }

    update(update) {
        if (update.docChanged || update.selectionSet || update.viewportChanged) {
            this.decorations = this.build(update.view);
        }
    }

    build(view) {
        const { doc } = view.state;
        const decorations = [];
        const stack = [];
        const openers = { '(': ')', '[': ']', '{': '}' };
        const closers = { ')': '(', ']': '[', '}': '{' };

        for (let pos = 0; pos < doc.length; pos++) {
            const ch = doc.sliceString(pos, pos + 1);
            if (openers[ch]) {
                stack.push({ from: pos, type: ch });
            } else if (closers[ch]) {
                const open = stack.pop();
                if (open && open.type === closers[ch]) {
                    const cls = classNames[stack.length % classNames.length];
                    decorations.push(
                        Decoration.mark({ class: cls }).range(open.from, open.from + 1),
                        Decoration.mark({ class: cls }).range(pos, pos + 1),
                    );
                }
            }
        }

        decorations.sort((a, b) => a.from - b.from || a.startSide - b.startSide);
        return Decoration.set(decorations);
    }
}, {
    decorations: v => v.decorations,
});

const theme = EditorView.baseTheme(
    Object.fromEntries(classNames.flatMap((cls, i) => [
        [`.${cls}`, { color: colors[i] }],
        [`.${cls} > span`, { color: colors[i] }],
    ]))
);

export function rainbowBrackets() {
    return [plugin, theme];
}
