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
  return <LayoutPanel width="calc(var(--spacing-8) * 10)" padding={5} isScrollable={false} className="control-sidebar" role="complementary" label="Controls">
    <VStack gap={5}>
      <HStack gap={2} vAlign="center"><Icon icon={AdjustmentsHorizontalIcon} color="accent"/><Heading level={2}>Controls</Heading></HStack>
      <Text id="engine-model" type="supporting" hidden>{''}</Text>
      <form id="control-form" noValidate>
        <fieldset id="control-fields" className="control-fields">
          <VStack gap={5}>
            <VStack id="scope-settings" gap={3} hidden>
              <Field label="Control scope" inputID="control-scope"><select id="control-scope" defaultValue="tibia"><option value="tibia">Tibia · 1 axis · 2 muscles</option><option value="lf15">Left front leg · 7 axes · 15 muscles</option></select></Field>
              <VStack id="multijoint-goal-settings" gap={3} hidden>
                <VStack id="goal-family-field" hidden><Field label="Target motion" inputID="goal-family"><select id="goal-family"/></Field></VStack>
                <VStack id="goal-joint-field"><Field label="Axis to scale" inputID="goal-joint" description="Other axes follow the reference trajectory."><select id="goal-joint"/></Field></VStack>
              </VStack>
            </VStack>
            <VStack gap={2}>
              <HStack hAlign="between" vAlign="center"><label htmlFor="goal-scale">Motion scale</label><Text type="supporting">Relative to reference</Text></HStack>
              <HStack gap={3} className="range-row"><input id="goal-scale" data-testid="goal-scale" type="range" min="0.8" max="1.2" step="0.005" defaultValue="0.975"/><HStack gap={1} className="number-with-unit"><input id="goal-scale-number" type="number" min="0.8" max="1.2" step="0.005" defaultValue="0.975" aria-label="Motion scale factor"/><Text type="supporting">×</Text></HStack></HStack>
              <HStack gap={2} className="preset-row" aria-label="Motion scale presets">{[['0.85','Smaller'], ['1','Reference'], ['1.15','Larger']].map(([scale,label]) => <Button key={scale} label={label} tooltip={`${scale}×`} size="sm" data-scale={scale} aria-pressed="false" className="preset-button"/>)}</HStack>
            </VStack>
            <VStack gap={2}><label htmlFor="initial-delta">Initial tibia angle offset</label><HStack gap={2} className="number-with-unit full-width"><input id="initial-delta" type="number" min="-5" max="5" step="0.5" defaultValue="0"/><Text type="supporting">°</Text></HStack><Text type="supporting">Shift the initial pose by up to 5° while keeping the same target.</Text></VStack>
            <fieldset className="mode-group"><legend>Run mode</legend><VStack gap={2}>
              {[['fast','Fast prediction','Predict once and verify in simulation'], ['auto','Auto refine','Refine if error exceeds the threshold'], ['refine','Refine input','Optimize up to the step limit'], ['closed_loop','Joint feedback','Correct pose and velocity errors every 2 ms']].map(([value,title,caption]) => <label className="mode-option" key={value} hidden={value === 'closed_loop'}><input id={`mode-${value}`} type="radio" name="mode" value={value} defaultChecked={value === 'fast'}/><VStack gap={0.5}><Text type="label">{title}</Text><Text type="supporting">{caption}</Text></VStack></label>)}
            </VStack></fieldset>
            <details className="advanced-settings"><summary>Advanced settings</summary><VStack gap={4} paddingBlockStart={4}>
              <VStack gap={2}><label htmlFor="threshold">Joint error threshold (°)</label><input id="threshold" type="number" min="0.01" max="180" step="any" defaultValue="5.73"/><Text type="supporting">Default threshold is about 5.73°.</Text></VStack>
              <VStack gap={2}><label htmlFor="correction-steps">Max refinement steps</label><input id="correction-steps" type="number" min="0" max="20" step="1" defaultValue="3" disabled/><Text id="correction-help" type="supporting">Fast prediction skips refinement.</Text></VStack>
              <VStack id="feedback-settings" gap={3} hidden>
                <Field label="Position gain (Kp)" inputID="feedback-kp"><input id="feedback-kp" type="number" min="0" step="any" defaultValue="20"/></Field>
                <Field label="Velocity gain (Kd)" inputID="feedback-kd"><input id="feedback-kd" type="number" min="0" step="any" defaultValue="0.3"/></Field>
              </VStack>
              <VStack hidden id="bio-settings"><Field label="Spike replay (experimental)" inputID="bio-mode"><select id="bio-mode"/></Field></VStack>
            </VStack></details>
            <label className="checkbox-row"><input id="render-video" type="checkbox" defaultChecked/><Text type="body">Render top-view video</Text></label>
          </VStack>
        </fieldset>
      </form>
      <VStack gap={2} className="run-action"><Text id="form-error" className="inline-error" role="alert" hidden>{''}</Text><Button id="run-button" data-testid="run-button" className="run-button" label="Run and verify" type="submit" form="control-form" variant="primary" width="100%" icon={<Icon icon={PlayIcon}/>}><Text id="run-button-label" type="inherit">Run and verify</Text></Button><Text type="supporting" className="run-help">Each result is verified in simulation.</Text></VStack>
      <VStack gap={2} className="scope-note"><Text type="label">Active scope</Text><Text id="scope-description" type="supporting">Left front tibia · 1 axis · 2 muscles · 13 teacher-driven channels</Text></VStack>
    </VStack>
  </LayoutPanel>;
}

function GoalPreview() {
  return <Card className="goal-preview" padding={4}>
    <HStack hAlign="between" vAlign="center" gap={3}><PanelTitle title="Target joint trajectory" caption="420 ms · 211 samples"/><HStack gap={1}><Button id="import-target" label="Import" size="sm" variant="ghost" icon={<Icon icon={DocumentArrowUpIcon}/>}/><Button id="export-target" label="Export" size="sm" variant="ghost" icon={<Icon icon={ArrowDownTrayIcon}/>}/></HStack></HStack>
    <VStack className="plot-wrap preview-plot"><canvas id="target-preview" data-testid="target-preview" role="img" aria-label="Target joint angle over 420 ms"/><Text id="preview-placeholder" className="plot-placeholder" type="supporting">Loading target trajectory…</Text></VStack>
    <HStack gap={3} wrap="wrap" hAlign="between"><Text id="goal-source" type="supporting">Preparing the default target…</Text><Text id="initial-angle-info" type="supporting">{''}</Text></HStack>
    <input id="goal-file" className="sr-only" type="file" accept="application/json,.json" tabIndex={-1} aria-label="Target trajectory JSON file"/>
  </Card>;
}

function JobProgress() {
  return <Card id="job-panel" className="job-panel" hidden padding={4} aria-labelledby="job-title">
    <HStack hAlign="between" gap={3}><Heading id="job-title" level={2}>Preparing job…</Heading><Text id="elapsed-time" type="code" hasTabularNumbers>00:00</Text></HStack>
    <ol id="job-steps" className="job-steps" aria-label="Job stages">{[['predicting','Predict rates'],['fast_verification','Verify'],['correcting','Refine'],['completed','Save']].map(([step,label],i) => <li data-step={step} key={step}><Text className="step-number">{i+1}</Text><Text type="supporting">{label}</Text></li>)}</ol>
    <Text id="job-detail" type="supporting" aria-live="polite">Waiting to run</Text><Text id="job-id" type="code" className="job-id">{''}</Text><Button id="retry-job" label="Retry status check" hidden size="sm"/>
  </Card>;
}

function EmptyStage() {
  return <Card id="empty-results" className="initial-stage" padding={0}>
    <HStack hAlign="between" vAlign="center" padding={4}><PanelTitle title="Top-view simulation"/><Text id="preview-scope" type="code" color="secondary">TOP / LF TIBIA</Text></HStack>
    <VStack className="empty-stage" vAlign="center" hAlign="center" gap={4}>
      <Icon icon={ViewfinderCircleIcon} size="lg" className="stage-target-icon" color="accent"/>
      <VStack hAlign="center" gap={2}><Heading level={2}>Ready to simulate</Heading><Text type="body" color="secondary">Choose a target and run the simulation.</Text></VStack>
      <HStack gap={3} className="empty-stage-meta"><Text type="code">420 ms</Text><Text type="supporting">·</Text><Text id="preview-pools" type="code">2 muscles</Text></HStack>
    </VStack>
    <HStack padding={4} hAlign="between" className="stage-footer"><Text type="supporting">Verified motion appears here.</Text><Text type="supporting">Waiting to run</Text></HStack>
  </Card>;
}

function Results() {
  return <VStack id="results" as="section" className="results-section" hidden gap={4} aria-labelledby="results-title">
    <HStack hAlign="between" vAlign="center" gap={3}><VStack gap={1}><Heading id="results-title" level={2}>Verification results</Heading><Text id="result-context" type="supporting">{''}</Text></VStack><Badge id="criterion-badge" label="Awaiting verification" variant="neutral" className="result-badge"/></HStack>
    <VStack id="correction-warning" data-testid="correction-warning" className="message warning-message" role="status" hidden gap={2} padding={4}><Text type="label">Refinement failed · using fast prediction</Text><Text id="correction-warning-text" type="supporting">{''}</Text><details><summary>Error details</summary><pre id="correction-error"/></details></VStack>
    <Text id="result-notice" className="message info-message" role="status" hidden>{''}</Text>
    <Text id="bio-result" className="message info-message" role="status" hidden>{''}</Text>
    <Grid columns={{minWidth: 135, max: 4}} gap={3} className="metrics-grid">
      <Card id="fast-metric" padding={4}><VStack gap={2}><Text type="supporting">Fast prediction RMSE</Text><Text id="fast-rmse" type="large" className="metric-value" hasTabularNumbers>—</Text><Text type="supporting">Joint angle error before refinement</Text></VStack></Card>
      <Card padding={4} className="metric-selected"><VStack gap={2}><Text type="supporting">Selected result RMSE</Text><Text id="selected-rmse" data-testid="result-rmse" type="large" className="metric-value" hasTabularNumbers>—</Text><Text id="threshold-caption" type="supporting">Lower is closer to the target.</Text></VStack></Card>
      <Card padding={4}><VStack gap={2}><Text type="supporting">Max absolute error</Text><Text id="max-error" type="large" className="metric-value" hasTabularNumbers>—</Text><Text id="max-error-caption" type="supporting">Largest deviation from the target</Text></VStack></Card>
      <Card padding={4}><VStack gap={2}><Text type="supporting">Selected input</Text><Text id="selected-stage" type="label" className="stage-value">—</Text><Text id="correction-caption" type="supporting">{''}</Text></VStack></Card>
    </Grid>
    <Card className="video-panel" padding={0}>
      <HStack hAlign="between" vAlign="center" padding={4} gap={3}><PanelTitle title="Top-view simulation" caption="Simulation replay with the selected input"/><Text type="code" color="secondary">0.25× PLAYBACK</Text></HStack>
      <VStack className="video-frame" vAlign="center" hAlign="center"><video id="result-video" data-testid="result-video" controls playsInline preload="metadata" aria-label="Top view of the fly's front leg under the selected input" hidden/><Text id="video-placeholder" type="supporting">Video appears here when ready.</Text></VStack>
      <VStack gap={2} padding={4} className="timeline-controls"><HStack hAlign="between"><label htmlFor="time-cursor">Time in motion</label><output id="time-output" htmlFor="time-cursor">0ms</output></HStack><input id="time-cursor" type="range" min="0" max="420" step="2" defaultValue="0"/><Text id="cursor-readout" className="cursor-readout" type="supporting">Scrub with the slider or video controls.</Text></VStack>
    </Card>
    <HStack id="multijoint-result-settings" vAlign="end" gap={3} wrap="wrap" hidden>
      <Field label="Joint axis to plot" inputID="result-joint"><select id="result-joint"/></Field>
      <Field label="Muscle to plot" inputID="result-muscle"><select id="result-muscle"/></Field>
      <Text id="result-joint-error" type="supporting">{''}</Text>
    </HStack>
    <Grid columns={{minWidth: 300, max: 2}} gap={4} className="result-charts">
      <Card padding={4}><HStack hAlign="between" gap={3}><PanelTitle title="Joint angles"/><HStack gap={3} className="legend"><Text className="legend-target" type="supporting">Target</Text><Text className="legend-actual" type="supporting">Actual</Text></HStack></HStack><VStack className="plot-wrap result-plot"><canvas id="angle-chart" data-testid="angle-chart" role="img" aria-label="Comparison of target and actual joint angles"/></VStack></Card>
      <Card padding={4}><HStack hAlign="between" gap={3}><VStack gap={1}><Heading id="rate-chart-title" level={2}>Predicted firing rates</Heading><Text id="rate-chart-detail" type="supporting" hidden>{''}</Text></VStack><HStack gap={3} className="legend" id="rate-legend"><Text className="legend-flexor" type="supporting">Flexor</Text><Text className="legend-extensor" type="supporting">Extensor</Text></HStack></HStack><VStack className="plot-wrap result-plot"><canvas id="rate-chart" data-testid="rate-chart" role="img" aria-label="Continuous firing rates by muscle group"/></VStack></Card>
    </Grid>
    <HStack gap={4} hAlign="between" wrap="wrap" className="result-footer"><Text type="supporting" className="result-explanation" id="result-explanation">RMSE measures joint angle error over 211 samples.</Text><HStack gap={2} wrap="wrap"><a id="protocol-link" href="about:blank" className="download-link" target="_blank" rel="noopener noreferrer" hidden><Text type="label">Protocol JSON</Text></a><a id="bundle-download" href="about:blank" className="download-link" download hidden><Icon icon={ArrowDownTrayIcon} size="sm"/><Text type="label">Download results</Text></a></HStack></HStack>
  </VStack>;
}

export function App() {
  useEffect(() => { startRateControl(); }, []);
  return <Theme theme={themes[defaultThemeSlug]} mode="dark">
    <AppShell height="auto" mobileNav={false} variant="section" topNav={<HStack className="engine-header" vAlign="center" hAlign="between" gap={3} paddingInline={5} paddingBlock={3}><HStack gap={3} vAlign="center"><Icon icon={CubeTransparentIcon} color="accent" size="lg"/><Text type="label" className="brand-name">REvFLY</Text><Text type="code" className="brand-version">CONTROL STUDIO</Text></HStack><HStack gap={4} vAlign="center"><a href="/openapi.json" target="_blank" rel="noreferrer" className="api-link">API spec <Icon icon={ArrowTopRightOnSquareIcon} size="xsm"/></a><HStack id="connection-status" className="connection-status" role="status" data-state="loading" gap={2}><Text className="status-dot" aria-hidden="true">●</Text><Text type="supporting">Checking connection…</Text></HStack></HStack></HStack>}>
      <Layout height="auto" start={<ControlPanel/>} content={<LayoutContent padding={5} isScrollable={false} className="workspace-content"><VStack gap={5}>
        <HStack className="page-heading" hAlign="between" vAlign="center" gap={3}><VStack gap={2}><Text type="supporting" className="eyebrow">MOTOR CONTROL WORKBENCH</Text><Heading level={1}>Set a target. See the motion.</Heading><Text color="secondary">Predict the firing rates for a target trajectory and verify the motion in simulation.</Text></VStack><Badge label="Simulation" variant="teal"/></HStack>
        <VStack id="page-error" className="message error-message" role="alert" hidden gap={2} padding={4}><Heading id="page-error-title" level={2}>Check the connection</Heading><Text id="page-error-text">{''}</Text><Button id="retry-connection" label="Reconnect" size="sm"/></VStack>
        <JobProgress/><EmptyStage/><Results/><GoalPreview/>
        <HStack as="footer" className="page-footer" hAlign="between" gap={3} wrap="wrap"><Text type="supporting">Target trajectory → trained model → firing rates → simulation</Text><Text type="supporting">420 ms</Text></HStack>
      </VStack></LayoutContent>}/>
      <Text id="status-announcement" className="sr-only" aria-live="polite" aria-atomic="true">{''}</Text>
    </AppShell>
  </Theme>;
}
