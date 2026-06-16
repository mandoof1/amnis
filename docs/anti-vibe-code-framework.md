# Anti-Vibe-Code Design Framework

**Professional Web Design Principles for Intentional, Polished Interfaces**

A synthesis of research from 42+ NLM sources, MoA parallel agent analysis, and primary research across NN/g, Material Design, Stripe, Linear, and industry design systems.

---

## Executive Summary

Professional web design isn't about taste — it's a system. Vibe-coding fails because it substitutes intuition for structure. Every polished site you've ever admired (Stripe, Linear, Vercel, Apple) uses the same underlying machinery: modular scales, constraint-based grids, tokenized design decisions, and interaction patterns rooted in cognitive science.

The difference between vibe-coded and intentional design is measurable. Vibe-coded sites have inconsistent spacing, arbitrary font sizes, no focal point, flat hierarchy, and interactions that feel disconnected. Intentional sites feel inevitable — every element has a reason for existing where it does.

This framework gives you the machinery: type scales, spacing tokens, layout patterns, color systems, micro-interaction guidelines, and a concrete checklist to audit your work.

---

## 1. Typography Discipline

### 1.1 Modular Type Scales

Stop picking font sizes by eye. Use a modular scale — a mathematical ratio that generates harmonious sizes from a base value.

| Ratio Name | Value | Use Case |
|---|---|---|
| Minor Second | 1.067 | Dense dashboards, data-heavy UIs |
| Major Second | 1.125 | Compact content, sidebars |
| Minor Third | 1.250 | **Recommended for most UIs** |
| Major Third | 1.250 | Content sites, blogs |
| Perfect Fourth | 1.333 | Editorial, long-form reading |
| Golden Ratio | 1.618 | Landing pages, marketing hero sections |

**CSS Implementation:**

```css
:root {
  --font-size-base: 1rem;          /* Respects user browser prefs */
  --ratio: 1.250;                   /* Minor third — good default */

  --font-size-sm: calc(1rem / var(--ratio));       /* ~12.8px at 16px base */
  --font-size-base: 1rem;                           /* 16px */
  --font-size-md: calc(1rem * var(--ratio));        /* 20px */
  --font-size-lg: calc(1rem * var(--ratio) * var(--ratio)); /* 25px */
  --font-size-xl: calc(1rem * var(--ratio) * var(--ratio) * var(--ratio)); /* 31px */
  --font-size-2xl: calc(1rem * pow(var(--ratio), 4)); /* 39px */
  --font-size-3xl: calc(1rem * pow(var(--ratio), 5)); /* 49px */
}

/* Modern fluid with container queries */
.fluid-body {
  font-size: clamp(var(--font-size-base), 0.75rem + 2cqi, var(--font-size-md));
}

.fluid-heading {
  font-size: clamp(var(--font-size-md), 1rem + 3cqi, var(--font-size-3xl));
}
```

### 1.2 Readability Science

| Property | Rule | Why |
|---|---|---|
| Measure (line length) | 45–75 characters | Beyond 75, the eye struggles to track the next line. Below 45, reading feels choppy. |
| Line height | 1.4–1.6 (body), 1.0–1.2 (headings) | Tight headings look intentional. Loose body text breathes. |
| Vertical rhythm | Align to baseline grid (e.g., 24px increments) | Every element snaps to the same invisible grid — text doesn't "float" |
| Paragraph spacing | 0.5–1em between paragraphs, 2–3em between sections | Clear visual breaks without visual noise |

**CSS Vertical Rhythm Pattern:**

```css
.prose {
  --rhythm: 1.5rem; /* 24px at 16px base */

  font-size: var(--font-size-base);
  line-height: 1.5;
}

.prose p {
  margin-bottom: var(--rhythm);
}

.prose h2 {
  font-size: var(--font-size-xl);
  line-height: 1.2;
  margin-top: calc(var(--rhythm) * 2);
  margin-bottom: var(--rhythm);
}

.prose h3 {
  font-size: var(--font-size-lg);
  line-height: 1.3;
  margin-top: calc(var(--rhythm) * 1.5);
  margin-bottom: calc(var(--rhythm) * 0.5);
}
```

### 1.3 Typography Anti-Patterns

| Anti-Pattern | Vibe-Code Tell | Fix |
|---|---|---|
| Random sizes | Headers that don't relate to body size mathematically | Use a modular scale |
| Center-aligned body text | Every line starts at a different x position | Left-align body, center only short headlines |
| Font overload | 3+ font families on one page | Max 2 families (one display, one text) |
| Default system fonts | Looks like a 2010s forum | Choose intentional fonts (Inter, IBM Plex, DM Sans, etc.) |
| All-caps body text | Shouting at the user | Reserve all-caps for labels, buttons, overlines |
| < 1.4 line height on body | Dense wall of text that's hard to read | 1.4–1.6 minimum |

---

## 2. Layout & Visual Hierarchy

### 2.1 The Three Layout Regimes

**Boxed (content-width)**
- Content centered with `max-width`, auto margins
- Use for: documentation, long-form articles, legal pages
- Risk: feels claustrophobic if used for everything

**Full-bleed**
- Content stretches edge-to-edge
- Use for: hero sections, galleries, immersive experiences
- Risk: no anchor, disorienting

**Hybrid (professional standard)**
- Most content lives in a centered column, selective elements break out
- Use for: 90% of professional sites

**The Full-Bleed Hybrid Layout Pattern:**

```css
.full-bleed {
  display: grid;
  grid-template-columns:
    1fr
    min(65ch, 100% - 4rem)
    1fr;
}

.full-bleed > * {
  grid-column: 2;
}

.full-bleed > .breakout {
  grid-column: 1 / -1;
}

.full-bleed > .full-width {
  grid-column: 1 / -1;
  width: 100%;
}
```

### 2.2 The 8px Grid System

The industry standard (Apple HIG, Material Design, Linear, Stripe all use it). Every spacing value is a multiple of 4px (fine grid) or 8px (standard grid).

```css
:root {
  --space-1: 0.25rem;  /* 4px */
  --space-2: 0.5rem;   /* 8px */
  --space-3: 0.75rem;  /* 12px */
  --space-4: 1rem;     /* 16px */
  --space-5: 1.5rem;   /* 24px */
  --space-6: 2rem;     /* 32px */
  --space-7: 3rem;     /* 48px */
  --space-8: 4rem;     /* 64px  — macro whitespace */
  --space-9: 6rem;     /* 96px  — section gap */
  --space-10: 8rem;    /* 128px — major section gap */
}

/* External spacing ≥ internal spacing */
.card {
  padding: var(--space-5);    /* internal: 24px */
  margin-bottom: var(--space-6); /* external: 32px (bigger) */
}
```

### 2.3 Visual Hierarchy Principles

**The Pattern of Focal Points** (not F-pattern, not Z-pattern — those describe what happens when hierarchy is absent):

Users look at the most dominant element first. Design the focal point, and the scanning path follows.

**Techniques for creating hierarchy:**

| Technique | How | Example |
|---|---|---|
| Size contrast | Make one thing significantly larger | Hero heading 3× larger than anything else |
| Weight contrast | Heavy vs light within same size | Bold label + regular value on a dashboard |
| Color contrast | Accent color on the action, muted on everything else | Blue button, gray everything else |
| Whitespace isolation | Surround one element with space | Callout box with 64px margin around it |
| Position | Top-left gets 40% of visual attention | Put the most important action there |

**The Asymmetrical Balance Pattern (anti-vibe-code hero):**

```html
<section class="hero-split">
  <div class="hero-content">
    <h1>Big statement here</h1>
    <p>Supporting text with value prop</p>
    <a href="#" class="cta">Primary action</a>
  </div>
  <div class="hero-visual">
    <!-- Illustration, screenshot, or abstract pattern -->
  </div>
</section>
```

```css
.hero-split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-8);
  align-items: center;
  min-height: 70vh;
}

/* On mobile, stack */
@media (max-width: 768px) {
  .hero-split {
    grid-template-columns: 1fr;
    gap: var(--space-6);
  }
}
```

### 2.4 Layout Anti-Patterns

| Tell | Looks Like | Fix |
|---|---|---|
| "Tube effect" | Every section the same ~1200px width with nothing breaking out | Hybrid grid with selective breakout elements |
| Sloppy symmetry | Everything perfectly centered with no asymmetry | Use asymmetric balance — intentional imbalance signals sophistication |
| Equal-width sections | Every section has the same proportions | Vary column rations (1/3–2/3, 1/4–3/4) |
| No focal point | All elements compete for attention equally | Make one element dominant per viewport |
| Card grid for everything | Every section is "3 cards with icon + title + description" | Vary section types: full-width, split, gallery, data, narrative |
| 1200px+ body text | Lines too long to read comfortably | Constrain text to 65ch max-width |

---

## 3. Color Theory for UI

### 3.1 The Functional Palette

Professional color is systematic, not expressive. Define colors by function:

```css
:root {
  /* Neutrals — the vast majority of your UI */
  --color-surface: #ffffff;
  --color-surface-secondary: #f5f5f5;
  --color-surface-tertiary: #e8e8e8;

  /* Text */
  --color-ink: #1a1a1a;          /* Off-black, never #000 */
  --color-ink-secondary: #6b6b6b; /* Muted but WCAG AA compliant */
  --color-ink-tertiary: #9e9e9e; /* For placeholders, disabled */

  /* Structure */
  --color-border: #e0e0e0;
  --color-border-hover: #bdbdbd;

  /* Accent — 10% of the UI */
  --color-accent: #2563eb;       /* Blue — primary action */
  --color-accent-hover: #1d4ed8;
  --color-accent-soft: #eff6ff;  /* Tinted surface for selected states */

  /* Semantic */
  --color-success: #16a34a;
  --color-warning: #d97706;
  --color-error: #dc2626;
}
```

### 3.2 Color Distribution — The 60-30-10 Rule

| % | Role | What |
|---|---|---|
| 60% | Surface / Background | Dominant neutral tone |
| 30% | Ink, borders, structural colors | Text, lines, subtle fills |
| 10% | Accent | Buttons, links, active states |

### 3.3 Why Not Pure Black (#000)

Pure black (#000) on pure white (#fff) at a 21:1 contrast ratio sounds ideal but causes **halation** — text edges appear to bleed into the background for many readers, especially with astigmatism (affects ~30% of the population).

**Use off-black and off-white instead:**

```css
/* Light mode */
--color-ink: #1a1a1a;           /* instead of #000 */
--color-surface: #ffffff;        /* white is fine — the surface */

/* Dark mode — critical: avoid #000 backgrounds */
--color-ink: #e4e4e7;           /* instead of #fff */
--color-surface: #18181b;       /* instead of #000 */
--color-surface-secondary: #27272a;
```

### 3.4 Modern Color with OKLCH

Traditional hex/RGB doesn't match human perception. OKLCH does — two colors with the same OKLCH lightness have the same perceived brightness, regardless of hue.

```css
/* L: lightness (0-100), C: chroma (0-0.4), H: hue (0-360) */
--color-blue-500: oklch(0.62 0.22 255);
--color-blue-600: oklch(0.55 0.24 255);
--color-blue-700: oklch(0.48 0.22 255);

/* Perceived lightness is consistent — scales predictably */
```

### 3.5 Accessibility Minimums

| Text Type | Size | Contrast Min |
|---|---|---|
| Body text | < 18px normal / < 14px bold | 4.5:1 (WCAG AA) |
| Large text | ≥ 18px bold / ≥ 24px normal | 3:1 (WCAG AA) |
| UI components | Any | 3:1 against adjacent colors |

---

## 4. Design Tokens — The Consistency Machine

### 4.1 Token Architecture

Design tokens prevent design drift by replacing hardcoded values with semantic variables. Three tiers:

```css
/* Tier 1: Reference Tokens (Options) — raw palette */
--ref-palette-blue-50: #eff6ff;
--ref-palette-blue-100: #dbeafe;
--ref-palette-blue-500: #3b82f6;
--ref-palette-gray-50: #f9fafb;
--ref-palette-gray-100: #f3f4f6;
--ref-palette-gray-900: #111827;

/* Tier 2: System Tokens (Decisions) — semantic roles */
--sys-color-surface: var(--ref-palette-gray-50);
--sys-color-text-primary: var(--ref-palette-gray-900);
--sys-color-text-secondary: #6b7280;
--sys-color-accent: var(--ref-palette-blue-500);
--sys-color-border: var(--ref-palette-gray-100);

/* Tier 3: Component Tokens (Scoped) — element-specific */
--comp-button-bg: var(--sys-color-accent);
--comp-button-text: #ffffff;
--comp-button-radius: var(--sys-radius-md);
```

### 4.2 Naming Convention (Nord Style)

```
{prefix}-{category}-{subcategory}-{property}

Examples:
n-color-surface-primary
n-space-component-gap
n-type-scale-heading-xl
n-radius-button-default
n-shadow-card-elevated
```

### 4.3 Token Categories You Need

| Category | Examples | Quantity |
|---|---|---|
| Color | surface, ink, accent, border, success, error | 12–20 |
| Spacing | 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 | 10 |
| Typography | font-size (sm–3xl), line-height, font-weight, font-family | 15–20 |
| Border radius | none, sm, md, lg, full | 5 |
| Shadow | sm, md, lg, xl, focus-ring | 5 |
| Motion duration | fast (100ms), normal (200ms), slow (500ms) | 3–5 |
| Motion easing | ease-out, ease-in-out, spring parameters | 3–5 |
| Breakpoints | sm, md, lg, xl | 4 |

### 4.4 Dark Mode with Tokens

Tokens make dark mode trivial — swap the reference values, keep the system tokens:

```css
@media (prefers-color-scheme: dark) {
  :root {
    --ref-palette-gray-50: #18181b;
    --ref-palette-gray-100: #27272a;
    --ref-palette-gray-900: #fafafa;
  }
  /* --sys-color-surface automatically updates */
  /* --comp-button-bg automatically updates */
}
```

---

## 5. Micro-interactions & Polish

### 5.1 The Interaction Contract

Every interaction is trigger → feedback. "Visual ghosting" happens when feedback is absent — the user acts and nothing happens.

**Always account for:**
| State | When | What to show |
|---|---|---|
| Default | Resting state | Normal appearance |
| Hover | Mouse over interactive element | Subtle feedback — lift, color shift, glow |
| Focus | Keyboard navigation | Visible focus ring (not `outline: none`) |
| Active | Mouse down | Brief compression (`scale: 0.97`) |
| Loading | Action in progress | Spinner, skeleton, or optimistic UI |
| Disabled | Action unavailable | Reduced opacity, no interactivity cursor |
| Error | Something went wrong | Red highlight, error message, color support |
| Empty | No data yet | Guidance for next action ("Create your first...") |

### 5.2 Animation Timing & Easing

```css
:root {
  /* Durations */
  --motion-micro: 100ms;     /* Hover, click feedback */
  --motion-fast: 150ms;      /* Tooltips, small reveals */
  --motion-normal: 200ms;    /* Toggle switches, menu opens */
  --motion-slow: 300ms;      /* Page transitions, modals */
  --motion-leisurely: 500ms; /* Loading state reveals */

  /* Easings */
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* Good: purposeful motion */
.button {
  transition: transform var(--motion-micro) var(--ease-out),
              background-color var(--motion-normal) var(--ease-out);
}

.button:hover {
  transform: translateY(-1px); /* Subtle lift — anticipates action */
}

.button:active {
  transform: scale(0.97); /* Physical compression — confirms action */
}

/* Bad: gratuitous motion — don't do this */
.badge {
  animation: spin 1s linear infinite; /* Spinning for no reason */
}
```

### 5.3 Micro-interaction Patterns

| Pattern | When to Use | Duration | Easing |
|---|---|---|---|
| Hover lift | Buttons, cards, links | 100ms | ease-out |
| Click compression | Buttons, toggles | 80ms | ease-in-out |
| Content reveal | Tooltips, dropdowns, modals | 150–200ms | ease-out |
| State toggle | Switches, checkboxes | 200ms | spring |
| Error shake | Wrong input, failed action | 200ms | custom |
| Loading skeleton | Content loading | 300ms stagger | ease-out |
| Page transition | Route changes | 200–300ms | ease-in-out |

### 5.4 The Difference Between Helpful and Harmful Animation

| Helpful (Intentional) | Harmful (Vibe-Coded) |
|---|---|
| Disappears after action completes | Loops forever |
| Lasts 100–300ms | Lasts 2+ seconds |
| Communicates state change | Exists "because animation looks cool" |
| Solves a problem (user needs to know X happened) | Solves nothing |
| Respects `prefers-reduced-motion` | Ignores motion preferences |
| Only animates what changed | Animates everything simultaneously |

---

## 6. Decision Tree: Is It Vibe-Coded?

For each element, ask these questions in order:

```
1. Does this element have an intentional size?
   ↓ Yes → It uses the type scale or spacing scale
   ↓ No  → Vibe-coded ✓

2. Does this element's position follow a grid?
   ↓ Yes → It aligns to the 8px grid or CSS grid
   ↓ No  → Vibe-coded ✓

3. Does this element's color serve a function?
   ↓ Yes → It's a semantic token (surface, ink, accent, border)
   ↓ No  → Vibe-coded ✓

4. Does this element respond to interaction?
   ↓ Yes → It has hover/focus/active states
   ↓ No  → Vibe-coded ✓

5. Is there an intentional focal point?
   ↓ Yes → One element clearly dominates
   ↓ No  → Vibe-coded ✓

6. Are the fonts intentional?
   ↓ Yes → Max 2 families, from a scale
   ↓ No  → Vibe-coded ✓

7. Is spacing consistent?
   ↓ Yes → Multiples of 4 or 8
   ↓ No  → Vibe-coded ✓
```

**Scoreboard:**
- **0-2 vibe-coded checks:** Professional polish
- **3-4 vibe-coded checks:** Needs a design pass
- **5-7 vibe-coded checks:** Full redesign needed

---

## 7. The Anti-Vibe-Code Audit Checklist

Use this to evaluate any page or component:

### Typography
- [ ] All font sizes derive from a modular scale (not eyeballed)
- [ ] Body line length ≤ 75 characters per line
- [ ] Body line height ≥ 1.4
- [ ] Max 2 font families on the page
- [ ] No center-aligned body text
- [ ] Headings sized on the same scale, not arbitrary
- [ ] Responsive typography uses `clamp()` with a `rem` anchor

### Spacing
- [ ] All spacing values are multiples of 4 or 8
- [ ] External spacing (margin) ≥ internal spacing (padding)
- [ ] Consistent vertical rhythm between sections
- [ ] Macro whitespace (64px+) separates distinct sections
- [ ] Micro whitespace (4-32px) creates relationships within components

### Layout
- [ ] A grid system governs the layout (not floating elements)
- [ ] One clear focal point per viewport
- [ ] Not every section is the same card-grid pattern
- [ ] Content is not stretched to 1200px+ width
- [ ] The hybrid full-bleed pattern or equivalent controls content width

### Color
- [ ] No pure #000 text on light backgrounds
- [ ] All text meets WCAG AA 4.5:1 minimum
- [ ] Colors serve functional roles (not just branding everywhere)
- [ ] The palette has: surface, ink, border, accent — at minimum
- [ ] Dark mode uses off-black backgrounds (not #000)

### Interactions
- [ ] Every interactive element has hover + focus states
- [ ] Click feedback exists (visual change within 100ms)
- [ ] Loading states show progress (don't just sit there)
- [ ] Empty states guide the user to a next action
- [ ] Error states are visible and actionable
- [ ] Animations are < 300ms and serve a purpose
- [ ] `prefers-reduced-motion` is respected

### System
- [ ] Design tokens exist (not hardcoded hex/value everywhere)
- [ ] Token naming follows a consistent convention
- [ ] Dark mode is a token swap, not a full restyle
- [ ] Components have a documented set of states
- [ ] Border radius is consistent across all elements
- [ ] Shadow/elevation values are consistent

---

## 8. Reference: Design Systems Analyzed

### Stripe
- **Typography:** Weight-300 elegance, generous letter-spacing on body
- **Grid:** Content is centered in wide but readable columns; visual elements occasionally break out
- **Color:** Signature purple gradients, but surfaces are mostly white/light gray
- **Interactions:** Buttons lift on hover, shadow transitions, smooth form focus states
- **Token structure:** 3-tier (reference → system → component)

### Linear
- **Typography:** Tight tracking, crisp sans-serif (Inter), mono accents on data
- **Grid:** Ultra-consistent 8px grid, everything aligns perfectly
- **Color:** Dark mode default, subtle purple accent, minimal palette
- **Interactions:** Smooth transitions on everything, spring-based easing
- **What makes it work:** Extreme consistency of a small design vocabulary applied everywhere

### Vercel
- **Typography:** Geist font system (geometric, tight), sharp size contrast
- **Grid:** Full-bleed layout with content-width text sections
- **Color:** Black, white, and one accent color (geometric blue)
- **Interactions:** Instant transitions, confident timing, no hesitation
- **What makes it work:** Minimalism with purpose — nothing exists without a reason

---

## 9. Recommended Tooling

| Tool | Purpose |
|---|---|
| [typescale.com](https://typescale.com) | Visualize and generate type scales |
| [gridlover.net](https://gridlover.net) | Generate vertical rhythm CSS |
| [coolors.co](https://coolors.co) | Palette exploration |
| [accessible-colors.com](https://accessible-colors.com) | WCAG contrast checking |
| [easing.dev](https://easing.dev) | Visualize and generate cubic-bezier curves |
| [precise-type.com](https://precise-type.com) | Precision type scale tool with CSS export |
| [acmefonts.app](https://acmefonts.app) | Font pairing experiments |

---

*This framework was researched and synthesized using NotebookLM (42 sources across NN/g, Material Design, CSS-Tricks, Smashing Magazine, Atlassian Design, Carbon Design, Stripe, Linear, and industry analysis) combined with MoA parallel agent research on layout, hierarchy, and design systems. The full source list is available in the NLM notebook.*
