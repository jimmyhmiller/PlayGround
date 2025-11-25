export const foo = (p, pattern, options = {}) => {
    if (!options.nocomment && pattern.charAt(0) === '#') {
        return false;
    }
    return true;
};
