import {defineTheme} from '@astryxdesign/core/theme';
import {neutralTheme} from '@astryxdesign/theme-neutral/built';

export const engineTheme = defineTheme({
  name: 'engine',
  extends: neutralTheme,
  tokens: {
    '--color-accent': '#43CFB4',
    '--color-accent-muted': '#43CFB422',
    '--color-on-accent': '#071A16',
    '--color-text-accent': '#68DDC6',
    '--color-icon-accent': '#68DDC6',
    '--color-background-body': '#101416',
    '--color-background-surface': '#171D20',
    '--color-background-card': '#1C2327',
    '--color-background-muted': '#13191C',
    '--color-background-popover': '#222B30',
    '--color-text-primary': '#E7EEEE',
    '--color-text-secondary': '#A0AFB5',
    '--color-text-placeholder': '#7D9098',
    '--color-border': '#303C42',
    '--color-border-emphasized': '#52636C',
    '--font-family-body': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    '--font-family-heading': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
});
