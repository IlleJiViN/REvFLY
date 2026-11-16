# Sensory Engine Control Studio

실제 `@astryxdesign/core` 0.7.15와 React 19를 사용하는 로컬 제어 화면입니다. Astryx IDE 템플릿의 패널 프레임을 기존 제어 화면에 맞춰 구성했습니다. Neutral에서 확장한 `engine` 테마를 빌드해 어두운 바탕과 청록 강조색을 적용합니다.

```bash
cd apps/control_ui
npm ci
npm run build
npm run typecheck
npx astryx doctor --built dist
```

`npm run build`가 기존 Python 서버가 읽는 세 파일을 생성합니다.

- `apps/templates/rate_control.html`: React 서버 렌더링 HTML.
- `static/rate_control.js`: React hydration, Astryx 컴포넌트, 기존 API 실행 코드가 들어간 하나의 자체 호스팅 번들.
- `static/rate_control.css`: Astryx 기본 스타일·빌드된 테마·작업 화면 스타일.

별도 프런트엔드 서버는 필요 없습니다. Python GUI 서버를 실행한 상태에서 `http://127.0.0.1:8505/`를 새로 고치면 됩니다. 폰트는 시스템 글꼴을 사용하고 CDN이나 원격 폰트 요청은 하지 않습니다.

`src/App.tsx`는 실제 Theme/AppShell/Layout/Card/Button/Badge/Text/Heading/Grid/Stack 컴포넌트로 화면을 만듭니다. 기존 입력·파일 불러오기·차트·영상은 원래의 native DOM 계약을 유지합니다. `src/rate-controller.js`는 hydration이 끝난 뒤 한 번만 연결됩니다. 입력은 uncontrolled이고, 실행 중 상태를 React가 별도로 덮어쓰지 않습니다. 캔버스 색은 다크 테마의 실제 CSS 색을 읽습니다.

이전의 MP4 검은 화면 문제를 피하도록 브라우저에서는 WebM을 우선 재생하고 PNG 포스터를 표시합니다. 요청 검증, 작업 폴링, 실패 표시, 세션 복원, 목표 내보내기, 결과 다운로드, 물리 시간 슬라이더는 같은 엔진 API를 사용합니다.

생성된 HTML/JS/CSS를 직접 수정하지 마세요. 화면은 `App.tsx`와 `workbench.css`, 동작은 `rate-controller.js`를 수정한 뒤 다시 빌드합니다. 팔레트는 `src/themes/engine/engineTheme.ts`에서 수정한 뒤 `npx astryx theme build src/themes/engine/engineTheme.ts`를 먼저 실행합니다.
