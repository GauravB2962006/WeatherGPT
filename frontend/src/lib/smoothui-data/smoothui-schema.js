// ---------------------------------------------------------------------------
// Validation constants
// ---------------------------------------------------------------------------

const VALID_CATEGORIES = [
  "basic-ui",
  "button",
  "text",
  "ai",
  "layout",
  "feedback",
  "data-display",
  "navigation",
  "other",
];

const VALID_COMPLEXITIES = [
  "simple",
  "moderate",
  "complex",
];

const VALID_ANIMATION_TYPES = [
  "spring",
  "tween",
  "gesture",
  "scroll",
  "none",
];

const isString = value => typeof value === "string";

const isBoolean = value => typeof value === "boolean";

const isStringArray = value => Array.isArray(value) && value.every(isString);

const isOneOf = (value, allowed) => isString(value) && (allowed).includes(value);

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/**
 * Validate and parse a raw `smoothui` field from a component package.json.
 *
 * Returns a discriminated union so callers can handle errors explicitly.
 *
 * @param raw - The unknown value read from `packageJson.smoothui`
 * @returns A `ParseResult` with either typed data or a list of errors
 */
export const parseSmoothUIMeta = raw => {
  const errors = [];

  if (raw === null || raw === undefined || typeof raw !== "object") {
    return {
      errors: [{ field: "smoothui", message: "Must be a non-null object" }],
      success: false,
    };
  }

  const obj = raw;

  // category — required
  if (!isOneOf(obj.category, VALID_CATEGORIES)) {
    errors.push({
      field: "category",
      message: `Must be one of: ${VALID_CATEGORIES.join(", ")}`,
    });
  }

  // tags — required string[]
  if (!isStringArray(obj.tags)) {
    errors.push({
      field: "tags",
      message: "Must be an array of strings",
    });
  }

  // complexity — required
  if (!isOneOf(obj.complexity, VALID_COMPLEXITIES)) {
    errors.push({
      field: "complexity",
      message: `Must be one of: ${VALID_COMPLEXITIES.join(", ")}`,
    });
  }

  // animationType — required
  if (!isOneOf(obj.animationType, VALID_ANIMATION_TYPES)) {
    errors.push({
      field: "animationType",
      message: `Must be one of: ${VALID_ANIMATION_TYPES.join(", ")}`,
    });
  }

  // useCases — required string[]
  if (!isStringArray(obj.useCases)) {
    errors.push({
      field: "useCases",
      message: "Must be an array of strings",
    });
  }

  // compositionHints — required string[]
  if (!isStringArray(obj.compositionHints)) {
    errors.push({
      field: "compositionHints",
      message: "Must be an array of strings",
    });
  }

  // hasReducedMotion — required boolean
  if (!isBoolean(obj.hasReducedMotion)) {
    errors.push({
      field: "hasReducedMotion",
      message: "Must be a boolean",
    });
  }

  if (errors.length > 0) {
    return { errors, success: false };
  }

  return {
    data: {
      animationType: obj.animationType,
      category: obj.category,
      complexity: obj.complexity,
      compositionHints: obj.compositionHints,
      hasReducedMotion: obj.hasReducedMotion,
      tags: obj.tags,
      useCases: obj.useCases,
    },
    success: true,
  };
};
