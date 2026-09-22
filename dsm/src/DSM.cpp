// Discontinuous Sound Modulator -- VCV Rack module. All logic lives in DsmCore.hpp
// (tested standalone); this file is the Rack binding: parameters, ports, lights, menu.
#include "plugin.hpp"
#include "DsmCore.hpp"
#include <osdialog.h>

// coloured knobs: Rack's Rogan1P shape with our own fills (res/Knob*.svg)
struct KnobPink : Rogan { KnobPink() { setSvg(Svg::load(asset::plugin(pluginInstance, "res/KnobPink.svg"))); bg->setSvg(Svg::load(asset::plugin(pluginInstance, "res/Knob_bg.svg"))); fg->setSvg(Svg::load(asset::plugin(pluginInstance, "res/KnobPink_fg.svg"))); } };
struct KnobYellow : Rogan { KnobYellow() { setSvg(Svg::load(asset::plugin(pluginInstance, "res/KnobYellow.svg"))); bg->setSvg(Svg::load(asset::plugin(pluginInstance, "res/Knob_bg.svg"))); fg->setSvg(Svg::load(asset::plugin(pluginInstance, "res/KnobYellow_fg.svg"))); } };
struct KnobCyan : Rogan { KnobCyan() { setSvg(Svg::load(asset::plugin(pluginInstance, "res/KnobCyan.svg"))); bg->setSvg(Svg::load(asset::plugin(pluginInstance, "res/Knob_bg.svg"))); fg->setSvg(Svg::load(asset::plugin(pluginInstance, "res/KnobCyan_fg.svg"))); } };

struct DSM : Module {
	enum ParamId {
		ON_PARAM, ENGINE_PARAM, MODE_PARAM, DELAY_PARAM, HOP_PARAM, RATE_PARAM, LOOP_PARAM, TS1_PARAM, VOLUME_PARAM,
		GX_PARAM, GZ_PARAM, AUTO_PARAM, RENDER_PARAM, NTRAJ_PARAM, TRAJ_PARAM, SYNC_PARAM, PARAMS_LEN };
	enum InputId { AUDIO_INPUT, ON_INPUT, INPUTS_LEN };
	enum OutputId { AUDIO_OUTPUT, OUTPUTS_LEN };
	enum LightId { REC_LIGHT, COMP_LIGHT, PLAY_LIGHT, ERR_LIGHT, LIGHTS_LEN };

	dsm::Core core;
	dsp::SchmittTrigger trajTrigger;
	bool onCv = false; float blink = 0.f;
	std::string spoolDir, exportDir;
	int threads = 4;

	DSM() {
		config(PARAMS_LEN, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);
		configSwitch(ON_PARAM, 0.f, 1.f, 0.f, "On / Off", {"Off", "On"});
		configSwitch(ENGINE_PARAM, 0.f, 1.f, 0.f, "Engine", {"B (C++)", "A (Python server)"});
		configSwitch(MODE_PARAM, 0.f, 1.f, 1.f, "Material", {"future", "past"});
		configParam(DELAY_PARAM, 1.f, 5.f, 3.f, "Delay (frames)"); getParamQuantity(DELAY_PARAM)->snapEnabled = true;
		configSwitch(HOP_PARAM, 0.f, 4.f, 1.f, "Hop", {"128", "256", "512", "1024", "2048"});
		configSwitch(RATE_PARAM, 0.f, 3.f, 0.f, "Model rate", {"/6", "/3", "/2", "/1"});
		configParam(LOOP_PARAM, 1.f, 10.f, 4.f, "Loop", " s");
		configParam(TS1_PARAM, 1.f, 4.f, 1.f, "TS1");
		configParam(VOLUME_PARAM, 0.f, 2.f, 1.f, "Volume");
		configParam(GX_PARAM, 0.1f, 1.2f, 0.5f, "gx");
		configParam(GZ_PARAM, 0.f, 1.5f, 0.2f, "gz");
		configSwitch(AUTO_PARAM, 0.f, 1.f, 1.f, "Couplings", {"knobs", "auto"});
		configSwitch(RENDER_PARAM, 0.f, 4.f, 0.f, "Render", {"undep", "dep0", "dep1", "classical FB", "pink noise"});
		configParam(NTRAJ_PARAM, 1.f, (float)dsm::MAX_TRAJ, 1.f, "Trajectories"); getParamQuantity(NTRAJ_PARAM)->snapEnabled = true;
		configButton(TRAJ_PARAM, "New set");
		configSwitch(SYNC_PARAM, 0.f, 1.f, 0.f, "Sync", {"off", "on: start the render when the input repeats the material"});
		configInput(AUDIO_INPUT, "Audio");
		configInput(ON_INPUT, "On gate (high = On)");
		configOutput(AUDIO_OUTPUT, "Audio");
		spoolDir = asset::user("dsm_spool");
		core.setSpoolDir(spoolDir);
		threads = std::max(1, (int)std::thread::hardware_concurrency() / 2 - 1);   // leave room for Rack and other instances
		core.setThreads(threads);
		core.setLogger([](const std::string& s) { INFO("DSM: %s", s.c_str()); });
	}

	void onSampleRateChange(const SampleRateChangeEvent& e) override { core.setSampleRate(e.sampleRate); }

	void process(const ProcessArgs& args) override {
		core.setSampleRate(args.sampleRate);
		static const int HOPS[5] = {128, 256, 512, 1024, 2048}; static const int DS[4] = {6, 3, 2, 1};
		dsm::CoreParams p = core.currentParams();
		p.delay = (int)std::round(params[DELAY_PARAM].getValue());
		p.H = HOPS[clamp((int)std::round(params[HOP_PARAM].getValue()), 0, 4)];
		p.D = DS[clamp((int)std::round(params[RATE_PARAM].getValue()), 0, 3)];
		p.t_l = std::round(params[LOOP_PARAM].getValue() * 100.f) / 100.f;
		p.autoCoupling = params[AUTO_PARAM].getValue() > 0.5f;
		p.gx = std::round(params[GX_PARAM].getValue() * 1000.f) / 1000.f;
		p.gz = std::round(params[GZ_PARAM].getValue() * 1000.f) / 1000.f;
		p.engine = params[ENGINE_PARAM].getValue() > 0.5f ? 1 : 0;
		p.past = params[MODE_PARAM].getValue() > 0.5f;
		p.render = clamp((int)std::round(params[RENDER_PARAM].getValue()), 0, 4);
		p.ntraj = clamp((int)std::round(params[NTRAJ_PARAM].getValue()), 1, dsm::MAX_TRAJ);
		core.setParams(p);
		core.setTs1(params[TS1_PARAM].getValue());
		core.setSync(params[SYNC_PARAM].getValue() > 0.5f);
		if (trajTrigger.process(params[TRAJ_PARAM].getValue())) core.newTrajectory();
		if (inputs[ON_INPUT].isConnected()) onCv = inputs[ON_INPUT].getVoltage() > 1.f;
		bool on = params[ON_PARAM].getValue() > 0.5f || (inputs[ON_INPUT].isConnected() && onCv);
		core.setOn(on);
		float in = inputs[AUDIO_INPUT].getVoltage() / 5.f;
		float y = core.process(in);
		outputs[AUDIO_OUTPUT].setVoltage(5.f * y * params[VOLUME_PARAM].getValue());
		// lights: REC = capturing (FUTURE), CMP = job running (engine B: brightness = progress; A/synth: blink),
		// PLAY = a loop plays, ERR = the last completed job failed or fell back (menu shows why)
		blink += args.sampleTime; if (blink >= 0.5f) blink -= 0.5f;
		dsm::Core::State s = core.state();
		bool busy = core.computing(); float pr = core.progress();
		lights[REC_LIGHT].setBrightness(s == dsm::Core::CAPTURING ? 1.f : 0.f);
		lights[COMP_LIGHT].setBrightness(!busy ? 0.f : (pr > 0.f ? 0.2f + 0.8f * pr : (blink < 0.25f ? 1.f : 0.15f)));
		lights[PLAY_LIGHT].setBrightness(core.syncWaiting() ? (blink < 0.25f ? 1.f : 0.15f) : (s == dsm::Core::PLAYING ? 1.f : 0.f));   // blinking green = render ready, waiting for the input to come round
		lights[ERR_LIGHT].setBrightness(core.hasError() ? 1.f : 0.f);
	}

	json_t* dataToJson() override {
		json_t* j = json_object(); json_object_set_new(j, "spoolDir", json_string(spoolDir.c_str())); json_object_set_new(j, "threads", json_integer(threads)); return j;
	}
	void dataFromJson(json_t* j) override {
		json_t* s = json_object_get(j, "spoolDir"); if (s) { spoolDir = json_string_value(s); core.setSpoolDir(spoolDir); }
		json_t* t = json_object_get(j, "threads"); if (t) { threads = clamp((int)json_integer_value(t), 1, 64); core.setThreads(threads); }
		// never auto-start on patch load: a saved ON would recompute at once (every instance, all threads)
		params[ON_PARAM].setValue(0.f);
	}
};

struct DSMWidget : ModuleWidget {
	DSMWidget(DSM* module) {
		setModule(module);
		setPanel(createPanel(asset::plugin(pluginInstance, "res/DSM.svg")));
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		float c1 = 30.f, c2 = 75.f, c3 = 120.f;
		// row 1 (red frame): ON, lights, MODE, ENGINE -- transport
		addParam(createParamCentered<VCVLatch>(Vec(c1, 58), module, DSM::ON_PARAM));
		addChild(createLightCentered<SmallLight<RedLight>>(Vec(c2 - 15, 52), module, DSM::REC_LIGHT));
		addChild(createLightCentered<SmallLight<YellowLight>>(Vec(c2 - 5, 52), module, DSM::COMP_LIGHT));
		addChild(createLightCentered<SmallLight<GreenLight>>(Vec(c2 + 5, 52), module, DSM::PLAY_LIGHT));
		addChild(createLightCentered<SmallLight<RedLight>>(Vec(c2 + 15, 52), module, DSM::ERR_LIGHT));
		addParam(createParamCentered<CKSS>(Vec(c2, 68), module, DSM::MODE_PARAM));
		addParam(createParamCentered<CKSS>(Vec(c3, 58), module, DSM::ENGINE_PARAM));
		addParam(createParamCentered<CKSS>(Vec(c3, 330 - 40), module, DSM::SYNC_PARAM));
		// row 2 (light blue): TS1, VOLUME, RENDER -- synthesis only, never a quantum computation
		addParam(createParamCentered<KnobCyan>(Vec(c1, 105), module, DSM::TS1_PARAM));
		addParam(createParamCentered<KnobCyan>(Vec(c2, 105), module, DSM::VOLUME_PARAM));
		addParam(createParamCentered<KnobCyan>(Vec(c3, 105), module, DSM::RENDER_PARAM));
		// row 3 (yellow): gx, gz, AUTO -- couplings
		addParam(createParamCentered<KnobYellow>(Vec(c1, 155), module, DSM::GX_PARAM));
		addParam(createParamCentered<KnobYellow>(Vec(c2, 155), module, DSM::GZ_PARAM));
		addParam(createParamCentered<CKSS>(Vec(c3, 155), module, DSM::AUTO_PARAM));
		// rows 4-5 (pink): DELAY, HOP, RATE / LOOP, TRAJECTORIES, NEW SET -- each change recomputes
		addParam(createParamCentered<KnobPink>(Vec(c1, 205), module, DSM::DELAY_PARAM));
		addParam(createParamCentered<KnobPink>(Vec(c2, 205), module, DSM::HOP_PARAM));
		addParam(createParamCentered<KnobPink>(Vec(c3, 205), module, DSM::RATE_PARAM));
		addParam(createParamCentered<KnobPink>(Vec(c1, 255), module, DSM::LOOP_PARAM));
		addParam(createParamCentered<KnobPink>(Vec(c2, 255), module, DSM::NTRAJ_PARAM));
		addParam(createParamCentered<VCVButton>(Vec(c3, 255), module, DSM::TRAJ_PARAM));
		// row 6: ports
		addInput(createInputCentered<PJ301MPort>(Vec(c1, 330), module, DSM::AUDIO_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(c2, 330), module, DSM::ON_INPUT));
		addOutput(createOutputCentered<PJ301MPort>(Vec(c3, 330), module, DSM::AUDIO_OUTPUT));
	}

	void appendContextMenu(Menu* menu) override {
		DSM* module = dynamic_cast<DSM*>(this->module);
		if (!module) return;
		menu->addChild(new MenuSeparator);
		dsm::CoreParams p = module->core.currentParams();
		double ticks = module->core.estimatedTicks(p);
		static const double MS_PER_TICK[6] = {0.003, 0.003, 0.005, 0.014, 0.05, 0.22};   // single-thread, engine B
		double est = ticks * MS_PER_TICK[clamp(p.delay, 1, 5)] / 1000.0 / module->threads;
		menu->addChild(createMenuLabel(string::f("Next quantum job: ~%.0f k ticks, ~%.0f s on %d threads (engine B)", ticks / 1000, est, module->threads)));
		menu->addChild(createSubmenuItem("Compute threads", std::to_string(module->threads), [=](Menu* sub) {
			int maxT = std::max(1, (int)std::thread::hardware_concurrency());
			for (int t = 1; t <= maxT; t++) sub->addChild(createCheckMenuItem(std::to_string(t), "", [=]() { return module->threads == t; }, [=]() { module->threads = t; module->core.setThreads(t); }));
		}));
		menu->addChild(new MenuSeparator);
		menu->addChild(createMenuLabel("Engine A spool: " + module->spoolDir));
		menu->addChild(createMenuItem("Set engine-A spool directory...", "", [=]() {
			char* c = osdialog_file(OSDIALOG_OPEN_DIR, module->spoolDir.c_str(), NULL, NULL);
			if (c) { module->spoolDir = c; module->core.setSpoolDir(module->spoolDir); free(c); }
		}));
		menu->addChild(createMenuItem("Export current render for verification...", "", [=]() {
			char* c = osdialog_file(OSDIALOG_OPEN_DIR, module->exportDir.empty() ? NULL : module->exportDir.c_str(), NULL, NULL);
			if (c) {
				std::string base = c; free(c);
				std::string dir = base + "/dsm_export_" + std::to_string((long long)system::getTime());
				bool ok = module->core.exportForVerification(dir);
				module->exportDir = base;
				if (!ok) osdialog_message(OSDIALOG_WARNING, OSDIALOG_OK, "Nothing to export: no render is playing.");
				else osdialog_message(OSDIALOG_INFO, OSDIALOG_OK, ("Exported to " + dir + "\nVerify with: python dsm_verify.py \"" + dir + "\"").c_str());
			}
		}));
		std::string err = module->core.lastError();
		if (!err.empty()) menu->addChild(createMenuLabel("Last error: " + err.substr(0, 90)));
		menu->addChild(createMenuLabel(string::f("Loop %.2f s, %d trajectories stored, state %d", module->core.loopSeconds(), module->core.storedTrajectories(), (int)module->core.state())));
	}
};

Model* modelDSM = createModel<DSM, DSMWidget>("DSM");
