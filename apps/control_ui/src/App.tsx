import {useEffect} from 'react';
import {AppShell} from '@astryxdesign/core/AppShell';
import {Theme} from '@astryxdesign/core/theme';
import {Layout, LayoutContent, LayoutPanel} from '@astryxdesign/core/Layout';
import {VStack} from '@astryxdesign/core/VStack';
import {HStack} from '@astryxdesign/core/HStack';
import {Grid} from '@astryxdesign/core/Grid';
import {Card} from '@astryxdesign/core/Card';
import {Button} from '@astryxdesign/core/Button';
import {Badge} from '@astryxdesign/core/Badge';
import {Text} from '@astryxdesign/core/Text';
import {Field} from '@astryxdesign/core/Field';
import {Heading} from '@astryxdesign/core/Heading';
import {Icon} from '@astryxdesign/core/Icon';
import {CubeTransparentIcon, PlayIcon, ArrowDownTrayIcon, AdjustmentsHorizontalIcon, ViewfinderCircleIcon, DocumentArrowUpIcon, ArrowTopRightOnSquareIcon} from '@heroicons/react/24/outline';
import {themes, defaultThemeSlug} from './astryx-themes';
import {startRateControl} from './rate-controller.js';

function PanelTitle({title, caption}: {title: string; caption?: string}) {
  return <VStack gap={1}><Heading level={2}>{title}</Heading>{caption && <Text type="supporting">{caption}</Text>}</VStack>;
}

function ControlPanel() {
  return <LayoutPanel width="calc(var(--spacing-8) * 10)" padding={5} isScrollable={false} className="control-sidebar" role="complementary" label="제어 설정">
    <VStack gap={5}>
      <HStack gap={2} vAlign="center"><Icon icon={AdjustmentsHorizontalIcon} color="accent"/><Heading level={2}>제어 설정</Heading></HStack>
      <Text id="engine-model" type="supporting" hidden>{''}</Text>
      <form id="control-form" noValidate>
        <fieldset id="control-fields" className="control-fields">
          <VStack gap={5}>
            <VStack id="scope-settings" gap={3} hidden>
              <Field label="제어 범위" inputID="control-scope"><select id="control-scope" defaultValue="tibia"><option value="tibia">경절 1축 · 근육 2개</option><option value="lf15">왼쪽 앞다리 7축 · 근육 15개</option></select></Field>
              <VStack id="multijoint-goal-settings" gap={3} hidden>
                <VStack id="goal-family-field" hidden><Field label="목표 동작" inputID="goal-family"><select id="goal-family"/></Field></VStack>
                <VStack id="goal-joint-field"><Field label="크기를 바꿀 축" inputID="goal-joint" description="다른 축은 기준 궤적을 유지합니다."><select id="goal-joint"/></Field></VStack>
              </VStack>
            </VStack>
            <VStack gap={2}>
              <HStack hAlign="between" vAlign="center"><label htmlFor="goal-scale">움직임 크기</label><Text type="supporting">원래 동작 대비</Text></HStack>
              <HStack gap={3} className="range-row"><input id="goal-scale" data-testid="goal-scale" type="range" min="0.8" max="1.2" step="0.005" defaultValue="0.975"/><HStack gap={1} className="number-with-unit"><input id="goal-scale-number" type="number" min="0.8" max="1.2" step="0.005" defaultValue="0.975" aria-label="움직임 크기 배율"/><Text type="supporting">×</Text></HStack></HStack>
              <HStack gap={2} className="preset-row" aria-label="움직임 크기 프리셋">{[['0.85','작게'], ['1','기본'], ['1.15','크게']].map(([scale,label]) => <Button key={scale} label={label} tooltip={`${scale}×`} size="sm" data-scale={scale} aria-pressed="false" className="preset-button"/>)}</HStack>
            </VStack>
            <VStack gap={2}><label htmlFor="initial-delta">경절 시작 각도 변화</label><HStack gap={2} className="number-with-unit full-width"><input id="initial-delta" type="number" min="-5" max="5" step="0.5" defaultValue="0"/><Text type="supporting">°</Text></HStack><Text type="supporting">목표는 유지하고 시작 자세만 ±5° 바꿉니다.</Text></VStack>
            <fieldset className="mode-group"><legend>실행 방식</legend><VStack gap={2}>
              {[['fast','빠른 생성','모델 1회 실행 후 물리 검증'], ['auto','필요할 때 보정','기준을 넘을 때만 입력 보정'], ['refine','입력 보정 실행','지정 횟수까지 최적화'], ['closed_loop','관절 피드백','2 ms마다 자세와 속도 오차를 보정']].map(([value,title,caption]) => <label className="mode-option" key={value} hidden={value === 'closed_loop'}><input id={`mode-${value}`} type="radio" name="mode" value={value} defaultChecked={value === 'fast'}/><VStack gap={0.5}><Text type="label">{title}</Text><Text type="supporting">{caption}</Text></VStack></label>)}
            </VStack></fieldset>
            <details className="advanced-settings"><summary>고급 설정</summary><VStack gap={4} paddingBlockStart={4}>
              <VStack gap={2}><label htmlFor="threshold">허용 관절 오차 · °</label><input id="threshold" type="number" min="0.01" max="180" step="any" defaultValue="5.73"/><Text type="supporting">기본 기준 약 5.73°</Text></VStack>
              <VStack gap={2}><label htmlFor="correction-steps">최대 보정 횟수</label><input id="correction-steps" type="number" min="0" max="20" step="1" defaultValue="3" disabled/><Text id="correction-help" type="supporting">빠른 생성에서는 보정하지 않아요.</Text></VStack>
              <VStack id="feedback-settings" gap={3} hidden>
                <Field label="자세 보정 강도 (Kp)" inputID="feedback-kp"><input id="feedback-kp" type="number" min="0" step="any" defaultValue="20"/></Field>
                <Field label="속도 보정 강도 (Kd)" inputID="feedback-kd"><input id="feedback-kd" type="number" min="0" step="any" defaultValue="0.3"/></Field>
              </VStack>
              <VStack hidden id="bio-settings"><Field label="스파이크 재생 방식 (실험)" inputID="bio-mode"><select id="bio-mode"/></Field></VStack>
            </VStack></details>
            <label className="checkbox-row"><input id="render-video" type="checkbox" defaultChecked/><Text type="body">탑뷰 영상 생성</Text></label>
          </VStack>
        </fieldset>
      </form>
      <VStack gap={2} className="run-action"><Text id="form-error" className="inline-error" role="alert" hidden>{''}</Text><Button id="run-button" data-testid="run-button" className="run-button" label="움직임 생성 및 검증" type="submit" form="control-form" variant="primary" width="100%" icon={<Icon icon={PlayIcon}/>}><Text id="run-button-label" type="inherit">움직임 생성 및 검증</Text></Button><Text type="supporting" className="run-help">모든 결과는 물리 시뮬레이션으로 검사합니다.</Text></VStack>
      <VStack gap={2} className="scope-note"><Text type="label">현재 실험 범위</Text><Text id="scope-description" type="supporting">왼쪽 앞다리 경절 1축 · 근육 2개 · 교사 보조 13개</Text></VStack>
    </VStack>
  </LayoutPanel>;
}

function GoalPreview() {
  return <Card className="goal-preview" padding={4}>
    <HStack hAlign="between" vAlign="center" gap={3}><PanelTitle title="목표 관절 궤적" caption="동작 0.42초 · 각도 211개"/><HStack gap={1}><Button id="import-target" label="불러오기" size="sm" variant="ghost" icon={<Icon icon={DocumentArrowUpIcon}/>}/><Button id="export-target" label="내보내기" size="sm" variant="ghost" icon={<Icon icon={ArrowDownTrayIcon}/>}/></HStack></HStack>
    <VStack className="plot-wrap preview-plot"><canvas id="target-preview" data-testid="target-preview" role="img" aria-label="0에서 420밀리초까지의 목표 관절각 그래프"/><Text id="preview-placeholder" className="plot-placeholder" type="supporting">목표 궤적을 불러오는 중입니다.</Text></VStack>
    <HStack gap={3} wrap="wrap" hAlign="between"><Text id="goal-source" type="supporting">기본 목표를 준비하고 있어요.</Text><Text id="initial-angle-info" type="supporting">{''}</Text></HStack>
    <input id="goal-file" className="sr-only" type="file" accept="application/json,.json" tabIndex={-1} aria-label="목표 궤적 JSON 파일"/>
  </Card>;
}

function JobProgress() {
  return <Card id="job-panel" className="job-panel" hidden padding={4} aria-labelledby="job-title">
    <HStack hAlign="between" gap={3}><Heading id="job-title" level={2}>작업을 준비하고 있어요.</Heading><Text id="elapsed-time" type="code" hasTabularNumbers>00:00</Text></HStack>
    <ol id="job-steps" className="job-steps" aria-label="실행 단계">{[['predicting','파형 생성'],['fast_verification','물리 검증'],['correcting','입력 보정'],['completed','결과 저장']].map(([step,label],i) => <li data-step={step} key={step}><Text className="step-number">{i+1}</Text><Text type="supporting">{label}</Text></li>)}</ol>
    <Text id="job-detail" type="supporting" aria-live="polite">실행 대기 중</Text><Text id="job-id" type="code" className="job-id">{''}</Text><Button id="retry-job" label="작업 상태 다시 조회" hidden size="sm"/>
  </Card>;
}

function EmptyStage() {
  return <Card id="empty-results" className="initial-stage" padding={0}>
    <HStack hAlign="between" vAlign="center" padding={4}><PanelTitle title="탑뷰 시뮬레이션"/><Text id="preview-scope" type="code" color="secondary">TOP / LF TIBIA</Text></HStack>
    <VStack className="empty-stage" vAlign="center" hAlign="center" gap={4}>
      <Icon icon={ViewfinderCircleIcon} size="lg" className="stage-target-icon" color="accent"/>
      <VStack hAlign="center" gap={2}><Heading level={2}>움직임을 준비해 볼까요?</Heading><Text type="body" color="secondary">왼쪽에서 목표를 정하고 실행하세요.</Text></VStack>
      <HStack gap={3} className="empty-stage-meta"><Text type="code">420 ms</Text><Text type="supporting">·</Text><Text id="preview-pools" type="code">근육 2개</Text></HStack>
    </VStack>
    <HStack padding={4} hAlign="between" className="stage-footer"><Text type="supporting">검증된 물리 결과가 여기에 표시됩니다.</Text><Text type="supporting">생성 대기</Text></HStack>
  </Card>;
}

function Results() {
  return <VStack id="results" as="section" className="results-section" hidden gap={4} aria-labelledby="results-title">
    <HStack hAlign="between" vAlign="center" gap={3}><VStack gap={1}><Heading id="results-title" level={2}>검증 결과</Heading><Text id="result-context" type="supporting">{''}</Text></VStack><Badge id="criterion-badge" label="평가 대기" variant="neutral" className="result-badge"/></HStack>
    <VStack id="correction-warning" data-testid="correction-warning" className="message warning-message" role="status" hidden gap={2} padding={4}><Text type="label">보정 실패 · 빠른 결과 유지</Text><Text id="correction-warning-text" type="supporting">{''}</Text><details><summary>오류 내용 확인</summary><pre id="correction-error"/></details></VStack>
    <Text id="result-notice" className="message info-message" role="status" hidden>{''}</Text>
    <Text id="bio-result" className="message info-message" role="status" hidden>{''}</Text>
    <Grid columns={{minWidth: 135, max: 4}} gap={3} className="metrics-grid">
      <Card id="fast-metric" padding={4}><VStack gap={2}><Text type="supporting">빠른 생성 RMSE</Text><Text id="fast-rmse" type="large" className="metric-value" hasTabularNumbers>—</Text><Text type="supporting">보정 전 관절각 오차</Text></VStack></Card>
      <Card padding={4} className="metric-selected"><VStack gap={2}><Text type="supporting">선택 결과 RMSE</Text><Text id="selected-rmse" data-testid="result-rmse" type="large" className="metric-value" hasTabularNumbers>—</Text><Text id="threshold-caption" type="supporting">낮을수록 목표에 가깝습니다.</Text></VStack></Card>
      <Card padding={4}><VStack gap={2}><Text type="supporting">최대 절대오차</Text><Text id="max-error" type="large" className="metric-value" hasTabularNumbers>—</Text><Text id="max-error-caption" type="supporting">가장 많이 벗어난 시점</Text></VStack></Card>
      <Card padding={4}><VStack gap={2}><Text type="supporting">선택된 입력</Text><Text id="selected-stage" type="label" className="stage-value">—</Text><Text id="correction-caption" type="supporting">{''}</Text></VStack></Card>
    </Grid>
    <Card className="video-panel" padding={0}>
      <HStack hAlign="between" vAlign="center" padding={4} gap={3}><PanelTitle title="탑뷰 시뮬레이션" caption="선택된 입력으로 재생한 물리 결과"/><Text type="code" color="secondary">0.25× PLAYBACK</Text></HStack>
      <VStack className="video-frame" vAlign="center" hAlign="center"><video id="result-video" data-testid="result-video" controls playsInline preload="metadata" aria-label="선택된 입력의 초파리 앞다리 탑뷰 영상" hidden/><Text id="video-placeholder" type="supporting">영상이 준비되면 여기에 표시됩니다.</Text></VStack>
      <VStack gap={2} padding={4} className="timeline-controls"><HStack hAlign="between"><label htmlFor="time-cursor">동작 시점</label><output id="time-output" htmlFor="time-cursor">0ms</output></HStack><input id="time-cursor" type="range" min="0" max="420" step="2" defaultValue="0"/><Text id="cursor-readout" className="cursor-readout" type="supporting">슬라이더 또는 영상으로 시점을 이동하세요.</Text></VStack>
    </Card>
    <HStack id="multijoint-result-settings" vAlign="end" gap={3} wrap="wrap" hidden>
      <Field label="그래프로 볼 축" inputID="result-joint"><select id="result-joint"/></Field>
      <Field label="그래프로 볼 근육" inputID="result-muscle"><select id="result-muscle"/></Field>
      <Text id="result-joint-error" type="supporting">{''}</Text>
    </HStack>
    <Grid columns={{minWidth: 300, max: 2}} gap={4} className="result-charts">
      <Card padding={4}><HStack hAlign="between" gap={3}><PanelTitle title="목표와 실제 관절각"/><HStack gap={3} className="legend"><Text className="legend-target" type="supporting">목표</Text><Text className="legend-actual" type="supporting">실제</Text></HStack></HStack><VStack className="plot-wrap result-plot"><canvas id="angle-chart" data-testid="angle-chart" role="img" aria-label="목표와 실제 관절각 비교 그래프"/></VStack></Card>
      <Card padding={4}><HStack hAlign="between" gap={3}><VStack gap={1}><Heading id="rate-chart-title" level={2}>생성한 연속 발화율</Heading><Text id="rate-chart-detail" type="supporting" hidden>{''}</Text></VStack><HStack gap={3} className="legend" id="rate-legend"><Text className="legend-flexor" type="supporting">굴근</Text><Text className="legend-extensor" type="supporting">신근</Text></HStack></HStack><VStack className="plot-wrap result-plot"><canvas id="rate-chart" data-testid="rate-chart" role="img" aria-label="근육 그룹의 연속 발화율 그래프"/></VStack></Card>
    </Grid>
    <HStack gap={4} hAlign="between" wrap="wrap" className="result-footer"><Text type="supporting" className="result-explanation" id="result-explanation">RMSE는 211개 시점의 관절각 오차입니다.</Text><HStack gap={2} wrap="wrap"><a id="protocol-link" href="about:blank" className="download-link" target="_blank" rel="noopener noreferrer" hidden><Text type="label">프로토콜 JSON</Text></a><a id="bundle-download" href="about:blank" className="download-link" download hidden><Icon icon={ArrowDownTrayIcon} size="sm"/><Text type="label">결과 다운로드</Text></a></HStack></HStack>
  </VStack>;
}

export function App() {
  useEffect(() => { startRateControl(); }, []);
  return <Theme theme={themes[defaultThemeSlug]} mode="dark">
    <AppShell height="auto" mobileNav={false} variant="section" topNav={<HStack className="engine-header" vAlign="center" hAlign="between" gap={3} paddingInline={5} paddingBlock={3}><HStack gap={3} vAlign="center"><Icon icon={CubeTransparentIcon} color="accent" size="lg"/><Text type="label" className="brand-name">Sensory Engine</Text><Text type="code" className="brand-version">CONTROL STUDIO</Text></HStack><HStack gap={4} vAlign="center"><a href="/openapi.json" target="_blank" rel="noreferrer" className="api-link">API 명세 <Icon icon={ArrowTopRightOnSquareIcon} size="xsm"/></a><HStack id="connection-status" className="connection-status" role="status" data-state="loading" gap={2}><Text className="status-dot" aria-hidden="true">●</Text><Text type="supporting">연결 확인 중</Text></HStack></HStack></HStack>}>
      <Layout height="auto" start={<ControlPanel/>} content={<LayoutContent padding={5} isScrollable={false} className="workspace-content"><VStack gap={5}>
        <HStack className="page-heading" hAlign="between" vAlign="center" gap={3}><VStack gap={2}><Text type="supporting" className="eyebrow">MOTOR CONTROL WORKBENCH</Text><Heading level={1}>목표를 정하고, 움직임을 확인하세요.</Heading><Text color="secondary">목표 궤적에서 연속 발화율을 생성하고 물리로 검증합니다.</Text></VStack><Badge label="시뮬레이션" variant="teal"/></HStack>
        <VStack id="page-error" className="message error-message" role="alert" hidden gap={2} padding={4}><Heading id="page-error-title" level={2}>연결을 확인해 주세요.</Heading><Text id="page-error-text">{''}</Text><Button id="retry-connection" label="연결 다시 확인" size="sm"/></VStack>
        <JobProgress/><EmptyStage/><Results/><GoalPreview/>
        <HStack as="footer" className="page-footer" hAlign="between" gap={3} wrap="wrap"><Text type="supporting">목표 궤적 → 학습 모델 → 연속 발화율 → 물리 검증</Text><Text type="supporting">0.42초</Text></HStack>
      </VStack></LayoutContent>}/>
      <Text id="status-announcement" className="sr-only" aria-live="polite" aria-atomic="true">{''}</Text>
    </AppShell>
  </Theme>;
}
