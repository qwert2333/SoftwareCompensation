#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TF1.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TLine.h"
#include "TPad.h"
#include "TProfile.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

namespace {

const TString kRunDir = "/Users/fangyi/WorkingArea/SoftwareCompensation";
const TString kOutputDir = kRunDir + "/outputs_dual_readout/standard_dual_readout";
const int kNEnergy = 6;
double kEnergyGeV[kNEnergy] = {5., 10., 20., 30., 40., 50.};

struct SampleInfo {
  TString fileName;
  double energyGeV = 0.;
};

struct StructureInfo {
  TString label;
  TString directory;
  std::vector<SampleInfo> electronSamples;
  std::vector<SampleInfo> pionSamples;
};

struct Calibration {
  double scintMeVPerReadout = 0.;
  double cherenkovMeVPerReadout = 0.;
};

struct ResolutionResult {
  double energyGeV = 0.;
  int entries = 0;
  double meanGeV = 0.;
  double mpvGeV = 0.;
  double sigma68GeV = 0.;
  double resolutionPct = 0.;
  double resolutionErrPct = 0.;
};

struct ReconstructionResults {
  ResolutionResult scint;
  ResolutionResult cherenkov;
  ResolutionResult dualReadout;
  double meanSGeV = 0.;
  double meanCGeV = 0.;
  double sigmaSGeV = 0.;
  double sigmaCGeV = 0.;
  double covSCGeV2 = 0.;
  double rhoSC = 0.;
  double chi = 0.;
  double femMean = 0.;
  double femRms = 0.;
  double sigmaDRRefGeV = 0.;
  double resolutionDRRefPct = 0.;
  double resolutionDRRefErrPct = 0.;
};

struct HoverEResult {
  double energyGeV = 0.;
  int entries = 0;
  double fitFemMin = 0.;
  double fitFemMax = 0.;
  double hoverES = 0.;
  double hoverESErr = 0.;
  double hoverEC = 0.;
  double hoverECErr = 0.;
  double chi2S = 0.;
  double ndfS = 0.;
  double chi2C = 0.;
  double ndfC = 0.;
};

struct HoverEFitFunction {
  double sIntercept = 0.;
  double sSlope = 0.;
  double cIntercept = 0.;
  double cSlope = 0.;
};

struct EdepLayerSummary {
  double energyGeV = 0.;
  int entries = 0;
  double scintMeanMeV = 0.;
  double scintRmsMeV = 0.;
  double scintMeanErrMeV = 0.;
  double cherenkovMeanMeV = 0.;
  double cherenkovRmsMeV = 0.;
  double cherenkovMeanErrMeV = 0.;
};

bool EndsWith(const std::string& text, const std::string& suffix)
{
  return text.size() >= suffix.size()
      && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool ParseEnergy(const std::string& path, const std::string& particle, double& energyGeV)
{
  const std::string tag = "_" + particle + "_";
  const auto particlePos = path.find(tag);
  if (particlePos == std::string::npos) return false;
  const auto energyStart = particlePos + tag.size();
  const auto energyEnd = path.find("GeV.root", energyStart);
  if (energyEnd == std::string::npos) return false;
  const std::string token = path.substr(energyStart, energyEnd - energyStart);
  try {
    std::size_t parsed = 0;
    energyGeV = std::stod(token, &parsed);
    if (parsed != token.size()) return false;
  } catch (...) {
    return false;
  }
  return true;
}

bool IsCalibrationEnergy(double energyGeV)
{
  for (double expected : kEnergyGeV) {
    if (std::fabs(energyGeV - expected) < 1e-9) return true;
  }
  return false;
}

bool IsIgnoredDirectory(const TString& label)
{
  return label == "plots"
      || label == "tables"
      || label == "summary_plots"
      || label == "summary_tables"
      || label == "Nph_resolution"
      || label == "Repeat_David"
      || label == "edep_layer_resolution"
      || label.BeginsWith("full_dualreadout")
      || label.BeginsWith("article_method")
      || label.BeginsWith("standard_dualreadout")
      || label.Contains("_analysis");
}

TString EnergyLabel(double energyGeV)
{
  if (std::fabs(energyGeV - int(energyGeV)) < 1e-9) return Form("%dGeV", int(energyGeV));
  return Form("%gGeV", energyGeV);
}

TString SafeName(TString text)
{
  text.ReplaceAll("-", "_");
  text.ReplaceAll(".", "p");
  text.ReplaceAll("/", "_");
  return text;
}

TString PlotDir(const StructureInfo& structure)
{
  return Form("%s/%s/plots", kOutputDir.Data(), structure.label.Data());
}

TString TableDir(const StructureInfo& structure)
{
  return Form("%s/%s/tables", kOutputDir.Data(), structure.label.Data());
}

TString NtupleDir(const StructureInfo& structure)
{
  return Form("%s/%s/ntuples", kOutputDir.Data(), structure.label.Data());
}

std::vector<StructureInfo> FindStructures()
{
  std::vector<StructureInfo> structures;
  for (const auto& entry : std::filesystem::directory_iterator(kRunDir.Data())) {
    if (!entry.is_directory()) continue;

    StructureInfo structure;
    structure.directory = entry.path().string();
    structure.label = entry.path().filename().string();
    if (IsIgnoredDirectory(structure.label)) continue;

    for (const auto& fileEntry : std::filesystem::directory_iterator(entry.path())) {
      if (!fileEntry.is_regular_file()) continue;
      const std::string path = fileEntry.path().string();
      if (!EndsWith(path, ".root")) continue;

      double energyGeV = 0.;
      if (ParseEnergy(path, "e-", energyGeV) && IsCalibrationEnergy(energyGeV)) {
        structure.electronSamples.push_back({path, energyGeV});
      } else if (ParseEnergy(path, "pi-", energyGeV) && IsCalibrationEnergy(energyGeV)) {
        structure.pionSamples.push_back({path, energyGeV});
      }
    }

    std::sort(structure.electronSamples.begin(), structure.electronSamples.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.energyGeV < rhs.energyGeV; });
    std::sort(structure.pionSamples.begin(), structure.pionSamples.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.energyGeV < rhs.energyGeV; });

    if (structure.electronSamples.size() == kNEnergy && structure.pionSamples.size() == kNEnergy) {
      structures.push_back(structure);
    }
  }

  std::sort(structures.begin(), structures.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.label < rhs.label; });
  return structures;
}

double Mean(const std::vector<double>& values)
{
  if (values.empty()) return 0.;
  double sum = 0.;
  for (double value : values) sum += value;
  return sum / values.size();
}

double Rms(const std::vector<double>& values, double mean)
{
  if (values.empty()) return 0.;
  double sum2 = 0.;
  for (double value : values) sum2 += (value - mean) * (value - mean);
  return std::sqrt(sum2 / values.size());
}

double Covariance(const std::vector<double>& lhs, const std::vector<double>& rhs,
                  double lhsMean, double rhsMean)
{
  if (lhs.empty() || lhs.size() != rhs.size()) return 0.;
  double sum = 0.;
  for (size_t i = 0; i < lhs.size(); ++i) {
    sum += (lhs[i] - lhsMean) * (rhs[i] - rhsMean);
  }
  return sum / lhs.size();
}

double Correlation(double covariance, double sigmaX, double sigmaY)
{
  return sigmaX > 0. && sigmaY > 0. ? covariance / (sigmaX * sigmaY) : 0.;
}

void DrawPionSCScatter(const StructureInfo& structure, const SampleInfo& sample,
                       const std::vector<double>& scintEnergyValues,
                       const std::vector<double>& cherenkovEnergyValues,
                       double covSCGeV2, double rhoSC)
{
  if (scintEnergyValues.empty() || scintEnergyValues.size() != cherenkovEnergyValues.size()) return;

  auto* graph = new TGraph(scintEnergyValues.size());
  double maxS = 0.;
  double maxC = 0.;
  for (size_t i = 0; i < scintEnergyValues.size(); ++i) {
    graph->SetPoint(i, scintEnergyValues[i], cherenkovEnergyValues[i]);
    maxS = std::max(maxS, scintEnergyValues[i]);
    maxC = std::max(maxC, cherenkovEnergyValues[i]);
  }
  graph->SetMarkerStyle(20);
  graph->SetMarkerSize(0.35);
  graph->SetMarkerColorAlpha(kBlue + 1, 0.28);

  auto* canvas = new TCanvas(Form("c_scatter_SC_%s_%s", SafeName(structure.label).Data(),
                                  EnergyLabel(sample.energyGeV).Data()),
                             "S vs C scatter", 850, 750);
  canvas->SetGrid();
  auto* frame = new TH1D(Form("frame_scatter_SC_%s_%s", SafeName(structure.label).Data(),
                              EnergyLabel(sample.energyGeV).Data()),
                         "", 100, 0., 1.15 * std::max(1., maxS));
  frame->SetDirectory(nullptr);
  frame->SetMinimum(0.);
  frame->SetMaximum(1.15 * std::max(1., maxC));
  frame->GetXaxis()->SetTitle("E_{S} [GeV, EM scale]");
  frame->GetYaxis()->SetTitle("E_{C} [GeV, EM scale]");
  frame->Draw();
  graph->Draw("P same");

  auto* label = new TLatex();
  label->SetNDC();
  label->SetTextFont(42);
  label->SetTextSize(0.038);
  label->DrawLatex(0.16, 0.86, Form("%s, #pi^{-} %s", structure.label.Data(),
                                     EnergyLabel(sample.energyGeV).Data()));
  label->DrawLatex(0.16, 0.80, Form("cov(S,C) = %.4g GeV^{2}", covSCGeV2));
  label->DrawLatex(0.16, 0.74, Form("#rho_{SC} = %.4f", rhoSC));

  canvas->SaveAs(Form("%s/pion_SC_scatter_%s.pdf", PlotDir(structure).Data(),
                      EnergyLabel(sample.energyGeV).Data()));
  canvas->SaveAs(Form("%s/pion_SC_scatter_%s.png", PlotDir(structure).Data(),
                      EnergyLabel(sample.energyGeV).Data()));

  delete label;
  delete frame;
  delete canvas;
  delete graph;
}

bool ReadChannelValues(const SampleInfo& sample, std::vector<double>& scint, std::vector<double>& cherenkov,
                       std::vector<double>* truth = nullptr)
{
  std::unique_ptr<TFile> file(TFile::Open(sample.fileName, "READ"));
  if (!file || file->IsZombie()) {
    std::cerr << "ERROR: cannot open " << sample.fileName << std::endl;
    return false;
  }
  auto* tree = dynamic_cast<TTree*>(file->Get("eventTree"));
  if (!tree) {
    std::cerr << "ERROR: cannot find eventTree in " << sample.fileName << std::endl;
    return false;
  }

  int scintReadout = 0;
  int cherenkovReadout = 0;
  double mcTruthEnergy = 0.;
  tree->SetBranchAddress("counter_Scintillation_ScintLayer", &scintReadout);
  tree->SetBranchAddress("counter_Cerenkov_CherenkovLayer", &cherenkovReadout);
  if (truth) tree->SetBranchAddress("MCtruth_energy", &mcTruthEnergy);

  scint.clear();
  cherenkov.clear();
  if (truth) truth->clear();
  scint.reserve(tree->GetEntries());
  cherenkov.reserve(tree->GetEntries());
  if (truth) truth->reserve(tree->GetEntries());

  for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
    tree->GetEntry(entry);
    scint.push_back(scintReadout);
    cherenkov.push_back(cherenkovReadout);
    if (truth) truth->push_back(mcTruthEnergy);
  }
  return true;
}

bool ReadEdepLayerValues(const SampleInfo& sample, std::vector<double>& scintEdepMeV,
                         std::vector<double>& cherenkovEdepMeV)
{
  std::unique_ptr<TFile> file(TFile::Open(sample.fileName, "READ"));
  if (!file || file->IsZombie()) {
    std::cerr << "ERROR: cannot open " << sample.fileName << std::endl;
    return false;
  }
  auto* tree = dynamic_cast<TTree*>(file->Get("eventTree"));
  if (!tree) {
    std::cerr << "ERROR: cannot find eventTree in " << sample.fileName << std::endl;
    return false;
  }

  double scintEdep = 0.;
  double cherenkovEdep = 0.;
  tree->SetBranchAddress("EdepScintLayer", &scintEdep);
  tree->SetBranchAddress("EdepCherenkovLayer", &cherenkovEdep);

  scintEdepMeV.clear();
  cherenkovEdepMeV.clear();
  scintEdepMeV.reserve(tree->GetEntries());
  cherenkovEdepMeV.reserve(tree->GetEntries());
  for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
    tree->GetEntry(entry);
    scintEdepMeV.push_back(scintEdep);
    cherenkovEdepMeV.push_back(cherenkovEdep);
  }
  return true;
}

EdepLayerSummary SummarizeEdepLayers(const SampleInfo& sample)
{
  EdepLayerSummary summary;
  summary.energyGeV = sample.energyGeV;

  std::vector<double> scintEdepMeV;
  std::vector<double> cherenkovEdepMeV;
  if (!ReadEdepLayerValues(sample, scintEdepMeV, cherenkovEdepMeV)) return summary;

  summary.entries = static_cast<int>(scintEdepMeV.size());
  summary.scintMeanMeV = Mean(scintEdepMeV);
  summary.scintRmsMeV = Rms(scintEdepMeV, summary.scintMeanMeV);
  summary.cherenkovMeanMeV = Mean(cherenkovEdepMeV);
  summary.cherenkovRmsMeV = Rms(cherenkovEdepMeV, summary.cherenkovMeanMeV);
  if (summary.entries > 0) {
    const double sqrtN = std::sqrt(double(summary.entries));
    summary.scintMeanErrMeV = summary.scintRmsMeV / sqrtN;
    summary.cherenkovMeanErrMeV = summary.cherenkovRmsMeV / sqrtN;
  }
  return summary;
}

void AnalyzeSamplingFractions(const StructureInfo& structure)
{
  std::vector<EdepLayerSummary> electronSummaries;
  std::vector<EdepLayerSummary> pionSummaries;
  electronSummaries.reserve(kNEnergy);
  pionSummaries.reserve(kNEnergy);
  for (const auto& sample : structure.electronSamples) {
    electronSummaries.push_back(SummarizeEdepLayers(sample));
  }
  for (const auto& sample : structure.pionSamples) {
    pionSummaries.push_back(SummarizeEdepLayers(sample));
  }

  gSystem->mkdir(TableDir(structure), true);
  std::ofstream pointCsv(Form("%s/edep_layer_sampling_points.csv", TableDir(structure).Data()));
  pointCsv << "structure,particle,energy_GeV,entries,"
           << "EdepScintLayer_mean_MeV,EdepScintLayer_rms_MeV,EdepScintLayer_mean_err_MeV,"
           << "EdepCherenkovLayer_mean_MeV,EdepCherenkovLayer_rms_MeV,EdepCherenkovLayer_mean_err_MeV,"
           << "EdepScintLayer_mean_over_Ebeam,EdepCherenkovLayer_mean_over_Ebeam\n";
  const auto writePoint = [&](const char* particle, const EdepLayerSummary& point) {
    const double beamMeV = 1000. * point.energyGeV;
    pointCsv << structure.label << "," << particle << "," << point.energyGeV << ","
             << point.entries << ","
             << point.scintMeanMeV << "," << point.scintRmsMeV << "," << point.scintMeanErrMeV << ","
             << point.cherenkovMeanMeV << "," << point.cherenkovRmsMeV << "," << point.cherenkovMeanErrMeV << ","
             << (beamMeV > 0. ? point.scintMeanMeV / beamMeV : 0.) << ","
             << (beamMeV > 0. ? point.cherenkovMeanMeV / beamMeV : 0.) << "\n";
  };
  for (const auto& point : electronSummaries) writePoint("e-", point);
  for (const auto& point : pionSummaries) writePoint("pi-", point);
  pointCsv.close();

  double energy[kNEnergy] = {0.};
  double zero[kNEnergy] = {0.};
  double eScint[kNEnergy] = {0.};
  double eScintErr[kNEnergy] = {0.};
  double eCherenkov[kNEnergy] = {0.};
  double eCherenkovErr[kNEnergy] = {0.};
  double piScint[kNEnergy] = {0.};
  double piScintErr[kNEnergy] = {0.};
  double piCherenkov[kNEnergy] = {0.};
  double piCherenkovErr[kNEnergy] = {0.};
  for (int i = 0; i < kNEnergy; ++i) {
    energy[i] = electronSummaries[i].energyGeV;
    eScint[i] = electronSummaries[i].scintMeanMeV;
    eScintErr[i] = electronSummaries[i].scintMeanErrMeV;
    eCherenkov[i] = electronSummaries[i].cherenkovMeanMeV;
    eCherenkovErr[i] = electronSummaries[i].cherenkovMeanErrMeV;
    piScint[i] = pionSummaries[i].scintMeanMeV;
    piScintErr[i] = pionSummaries[i].scintMeanErrMeV;
    piCherenkov[i] = pionSummaries[i].cherenkovMeanMeV;
    piCherenkovErr[i] = pionSummaries[i].cherenkovMeanErrMeV;
  }

  auto* graphEScint = new TGraphErrors(kNEnergy, energy, eScint, zero, eScintErr);
  auto* graphECherenkov = new TGraphErrors(kNEnergy, energy, eCherenkov, zero, eCherenkovErr);
  auto* graphPiScint = new TGraphErrors(kNEnergy, energy, piScint, zero, piScintErr);
  auto* graphPiCherenkov = new TGraphErrors(kNEnergy, energy, piCherenkov, zero, piCherenkovErr);

  auto setupGraph = [](TGraphErrors* graph, Color_t color, Style_t marker) {
    graph->SetMarkerColor(color);
    graph->SetLineColor(color);
    graph->SetMarkerStyle(marker);
    graph->SetMarkerSize(1.1);
  };
  setupGraph(graphEScint, kBlue + 1, 20);
  setupGraph(graphECherenkov, kAzure + 2, 24);
  setupGraph(graphPiScint, kRed + 1, 21);
  setupGraph(graphPiCherenkov, kOrange + 7, 25);

  auto makeFit = [&](const char* name, Color_t color) {
    auto* fit = new TF1(Form("%s_%s", name, SafeName(structure.label).Data()), "[0]*x", 0., 55.);
    fit->SetParName(0, "slope_MeV_per_GeV");
    fit->SetLineColor(color);
    fit->SetLineWidth(2);
    return fit;
  };
  auto* fitEScint = makeFit("fit_edep_e_scint", kBlue + 1);
  auto* fitECherenkov = makeFit("fit_edep_e_cherenkov", kAzure + 2);
  auto* fitPiScint = makeFit("fit_edep_pi_scint", kRed + 1);
  auto* fitPiCherenkov = makeFit("fit_edep_pi_cherenkov", kOrange + 7);
  graphEScint->Fit(fitEScint, "Q0");
  graphECherenkov->Fit(fitECherenkov, "Q0");
  graphPiScint->Fit(fitPiScint, "Q0");
  graphPiCherenkov->Fit(fitPiCherenkov, "Q0");

  std::ofstream fitCsv(Form("%s/edep_layer_sampling_fraction.csv", TableDir(structure).Data()));
  fitCsv << "structure,particle,component,branch,sampling_fraction_type,"
         << "slope_MeV_per_GeV,slope_err_MeV_per_GeV,sampling_fraction,sampling_fraction_err,"
         << "chi2,ndf,fit_model\n";
  const auto writeFit = [&](const char* particle, const char* component, const char* branch,
                            const char* type, TF1* fit) {
    fitCsv << structure.label << "," << particle << "," << component << "," << branch << "," << type << ","
           << fit->GetParameter(0) << "," << fit->GetParError(0) << ","
           << fit->GetParameter(0) / 1000. << "," << fit->GetParError(0) / 1000. << ","
           << fit->GetChisquare() << "," << fit->GetNDF() << ","
           << "\"mean_Edep_MeV=slope*Ebeam_GeV\"\n";
  };
  writeFit("e-", "scintillation_layer", "EdepScintLayer", "EM", fitEScint);
  writeFit("e-", "cherenkov_layer", "EdepCherenkovLayer", "EM", fitECherenkov);
  writeFit("pi-", "scintillation_layer", "EdepScintLayer", "hadronic_effective", fitPiScint);
  writeFit("pi-", "cherenkov_layer", "EdepCherenkovLayer", "hadronic_effective", fitPiCherenkov);
  fitCsv.close();

  gSystem->mkdir(PlotDir(structure), true);
  auto* canvas = new TCanvas(Form("c_edep_sampling_%s", SafeName(structure.label).Data()),
                             "sampling fraction from deposited energy", 950, 750);
  canvas->SetGrid();
  double maxY = 0.;
  for (int i = 0; i < kNEnergy; ++i) {
    maxY = std::max({maxY, eScint[i] + eScintErr[i], eCherenkov[i] + eCherenkovErr[i],
                     piScint[i] + piScintErr[i], piCherenkov[i] + piCherenkovErr[i]});
  }
  auto* frame = new TH1D(Form("frame_edep_sampling_%s", SafeName(structure.label).Data()), "", 100, 0., 55.);
  frame->SetDirectory(nullptr);
  frame->SetMinimum(0.);
  frame->SetMaximum(std::max(1., 1.25 * maxY));
  frame->GetXaxis()->SetTitle("Beam energy [GeV]");
  frame->GetYaxis()->SetTitle("Mean deposited energy in layer [MeV]");
  frame->Draw();

  graphEScint->Draw("P same");
  graphECherenkov->Draw("P same");
  graphPiScint->Draw("P same");
  graphPiCherenkov->Draw("P same");
  fitEScint->Draw("same");
  fitECherenkov->Draw("same");
  fitPiScint->Draw("same");
  fitPiCherenkov->Draw("same");

  auto* legend = new TLegend(0.15, 0.60, 0.6, 0.88);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(graphEScint, Form("e^{-} EdepScintLayer: f = %.4f", fitEScint->GetParameter(0) / 1000.), "lep");
  legend->AddEntry(graphECherenkov, Form("e^{-} EdepCherenkovLayer: f = %.4f", fitECherenkov->GetParameter(0) / 1000.), "lep");
  legend->AddEntry(graphPiScint, Form("#pi^{-} EdepScintLayer: f = %.4f", fitPiScint->GetParameter(0) / 1000.), "lep");
  legend->AddEntry(graphPiCherenkov, Form("#pi^{-} EdepCherenkovLayer: f = %.4f", fitPiCherenkov->GetParameter(0) / 1000.), "lep");
  legend->Draw();

  canvas->SaveAs(Form("%s/edep_layer_sampling_fraction.pdf", PlotDir(structure).Data()));
  canvas->SaveAs(Form("%s/edep_layer_sampling_fraction.png", PlotDir(structure).Data()));

  delete legend;
  delete frame;
  delete canvas;
  delete fitPiCherenkov;
  delete fitPiScint;
  delete fitECherenkov;
  delete fitEScint;
  delete graphPiCherenkov;
  delete graphPiScint;
  delete graphECherenkov;
  delete graphEScint;
}

ResolutionResult ComputeResolution(const std::vector<double>& energyValues, double energyGeV,
                                   const TString& histName)
{
  ResolutionResult result;
  result.energyGeV = energyGeV;
  result.entries = static_cast<int>(energyValues.size());
  if (energyValues.empty()) return result;

  auto values = energyValues;
  std::sort(values.begin(), values.end());
  const double minE = values.front();
  const double maxE = values.back();
  const double span = std::max(1e-9, maxE - minE);

  auto* hist = new TH1D(histName, "", 80, minE - 0.05 * span, maxE + 0.05 * span);
  hist->SetDirectory(nullptr);
  for (double value : energyValues) hist->Fill(value);

  const int nEntries = static_cast<int>(values.size());
  const int nWindow = std::max(1, int(std::round(0.68 * nEntries)));
  int bestStart = 0;
  double bestWidth = 1e99;
  for (int i = 0; i + nWindow - 1 < nEntries; ++i) {
    const double width = values[i + nWindow - 1] - values[i];
    if (width < bestWidth) {
      bestWidth = width;
      bestStart = i;
    }
  }

  result.meanGeV = hist->GetMean();
  result.mpvGeV = hist->GetBinCenter(hist->GetMaximumBin());
  result.sigma68GeV = 0.5 * (values[bestStart + nWindow - 1] - values[bestStart]);
  result.resolutionPct = result.mpvGeV > 0. ? 100. * result.sigma68GeV / result.mpvGeV : 0.;
  result.resolutionErrPct = result.resolutionPct > 0.
                          ? result.resolutionPct / std::sqrt(2. * nEntries) : 0.;
  delete hist;
  return result;
}

Calibration CalibrateAndPlotElectrons(const StructureInfo& structure)
{
  double sumE2 = 0.;
  double sumES = 0.;
  double sumEC = 0.;
  std::vector<std::vector<double>> scintRaw(kNEnergy);
  std::vector<std::vector<double>> cherenkovRaw(kNEnergy);
  std::vector<double> truthMeV(kNEnergy, 0.);

  for (int i = 0; i < kNEnergy; ++i) {
    std::vector<double> truthValues;
    ReadChannelValues(structure.electronSamples[i], scintRaw[i], cherenkovRaw[i], &truthValues);
    truthMeV[i] = Mean(truthValues);
    const double scintMean = Mean(scintRaw[i]);
    const double cherenkovMean = Mean(cherenkovRaw[i]);
    sumE2 += truthMeV[i] * truthMeV[i];
    sumES += truthMeV[i] * scintMean;
    sumEC += truthMeV[i] * cherenkovMean;
  }

  Calibration calibration;
  calibration.scintMeVPerReadout = sumES > 0. ? sumE2 / sumES : 0.;
  calibration.cherenkovMeVPerReadout = sumEC > 0. ? sumE2 / sumEC : 0.;

  std::vector<ResolutionResult> sResults;
  std::vector<ResolutionResult> cResults;
  sResults.reserve(kNEnergy);
  cResults.reserve(kNEnergy);

  for (int i = 0; i < kNEnergy; ++i) {
    std::vector<double> sEnergy;
    std::vector<double> cEnergy;
    sEnergy.reserve(scintRaw[i].size());
    cEnergy.reserve(cherenkovRaw[i].size());
    for (size_t entry = 0; entry < scintRaw[i].size(); ++entry) {
      sEnergy.push_back(scintRaw[i][entry] * calibration.scintMeVPerReadout / 1000.);
      cEnergy.push_back(cherenkovRaw[i][entry] * calibration.cherenkovMeVPerReadout / 1000.);
    }
    sResults.push_back(ComputeResolution(sEnergy, structure.electronSamples[i].energyGeV,
                                         Form("h_electron_S_%s_%s", SafeName(structure.label).Data(),
                                              EnergyLabel(structure.electronSamples[i].energyGeV).Data())));
    cResults.push_back(ComputeResolution(cEnergy, structure.electronSamples[i].energyGeV,
                                         Form("h_electron_C_%s_%s", SafeName(structure.label).Data(),
                                              EnergyLabel(structure.electronSamples[i].energyGeV).Data())));
  }

  gSystem->mkdir(TableDir(structure), true);
  std::ofstream csv(Form("%s/electron_em_calibration.csv", TableDir(structure).Data()));
  csv << "structure,channel,branch,energy_GeV,entries,calibration_MeV_per_readout,"
      << "mean_GeV,mpv_GeV,sigma68_GeV,resolution_percent,resolution_error_percent\n";
  for (int i = 0; i < kNEnergy; ++i) {
    csv << structure.label << ",S,counter_Scintillation_ScintLayer," << sResults[i].energyGeV << ","
        << sResults[i].entries << "," << calibration.scintMeVPerReadout << ","
        << sResults[i].meanGeV << "," << sResults[i].mpvGeV << "," << sResults[i].sigma68GeV << ","
        << sResults[i].resolutionPct << "," << sResults[i].resolutionErrPct << "\n";
    csv << structure.label << ",C,counter_Cerenkov_CherenkovLayer," << cResults[i].energyGeV << ","
        << cResults[i].entries << "," << calibration.cherenkovMeVPerReadout << ","
        << cResults[i].meanGeV << "," << cResults[i].mpvGeV << "," << cResults[i].sigma68GeV << ","
        << cResults[i].resolutionPct << "," << cResults[i].resolutionErrPct << "\n";
  }
  csv.close();

  double energy[kNEnergy] = {0.};
  double zero[kNEnergy] = {0.};
  double sRes[kNEnergy] = {0.};
  double sErr[kNEnergy] = {0.};
  double cRes[kNEnergy] = {0.};
  double cErr[kNEnergy] = {0.};
  for (int i = 0; i < kNEnergy; ++i) {
    energy[i] = sResults[i].energyGeV;
    sRes[i] = sResults[i].resolutionPct;
    sErr[i] = sResults[i].resolutionErrPct;
    cRes[i] = cResults[i].resolutionPct;
    cErr[i] = cResults[i].resolutionErrPct;
  }

  gSystem->mkdir(PlotDir(structure), true);
  auto* canvas = new TCanvas(Form("c_electron_resolution_%s", SafeName(structure.label).Data()),
                             "electron EM calibration resolution", 900, 700);
  canvas->SetGrid();
  auto* frame = new TH1D(Form("frame_electron_resolution_%s", SafeName(structure.label).Data()),
                         "", 100, 0., 55.);
  frame->SetDirectory(nullptr);
  frame->SetMinimum(0.);
  frame->SetMaximum(20.);
  frame->GetXaxis()->SetTitle("Beam energy [GeV]");
  frame->GetYaxis()->SetTitle("#sigma_{68} / MPV [%]");
  frame->Draw();

  auto* graphS = new TGraphErrors(kNEnergy, energy, sRes, zero, sErr);
  graphS->SetLineColor(kBlue + 1);
  graphS->SetMarkerColor(kBlue + 1);
  graphS->SetMarkerStyle(20);
  graphS->Draw("P same");
  auto* graphC = new TGraphErrors(kNEnergy, energy, cRes, zero, cErr);
  graphC->SetLineColor(kRed + 1);
  graphC->SetMarkerColor(kRed + 1);
  graphC->SetMarkerStyle(21);
  graphC->Draw("P same");

  auto* legend = new TLegend(0.44, 0.72, 0.88, 0.88);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(graphS, Form("S calib = %.4g MeV/count", calibration.scintMeVPerReadout), "lep");
  legend->AddEntry(graphC, Form("C calib = %.4g MeV/count", calibration.cherenkovMeVPerReadout), "lep");
  legend->Draw();
  canvas->SaveAs(Form("%s/electron_em_resolution.pdf", PlotDir(structure).Data()));
  canvas->SaveAs(Form("%s/electron_em_resolution.png", PlotDir(structure).Data()));

  delete legend;
  delete graphC;
  delete graphS;
  delete frame;
  delete canvas;

  return calibration;
}

TF1* MakeHoverEFit(const TString& name, double fitMin, double fitMax)
{
  auto* fit = new TF1(name, "[0] + (1. - [0]) * x", fitMin, fitMax);
  fit->SetParName(0, "h/e");
  fit->SetParameter(0, 0.7);
  fit->SetParLimits(0, 0., 1.5);
  return fit;
}

double FitFemMin(double energyGeV)
{
  return energyGeV < 30. ? 0.10 : 0.30;
}

HoverEResult FitHoverEOneEnergy(const StructureInfo& structure, const SampleInfo& sample,
                                const Calibration& calibration)
{
  HoverEResult result;
  result.energyGeV = sample.energyGeV;
  result.fitFemMin = FitFemMin(sample.energyGeV);
  result.fitFemMax = 0.90;

  std::unique_ptr<TFile> file(TFile::Open(sample.fileName, "READ"));
  if (!file || file->IsZombie()) {
    std::cerr << "ERROR: cannot open " << sample.fileName << std::endl;
    return result;
  }
  auto* tree = dynamic_cast<TTree*>(file->Get("eventTree"));
  if (!tree) {
    std::cerr << "ERROR: cannot find eventTree in " << sample.fileName << std::endl;
    return result;
  }

  double truthEdepEM = 0.;
  double mcTruthEnergy = 0.;
  int scintReadout = 0;
  int cherenkovReadout = 0;
  tree->SetBranchAddress("truthEdepEM", &truthEdepEM);
  tree->SetBranchAddress("MCtruth_energy", &mcTruthEnergy);
  tree->SetBranchAddress("counter_Scintillation_ScintLayer", &scintReadout);
  tree->SetBranchAddress("counter_Cerenkov_CherenkovLayer", &cherenkovReadout);

  auto* profileS = new TProfile(Form("prof_S_%s_%s", SafeName(structure.label).Data(),
                                     EnergyLabel(sample.energyGeV).Data()), ";truth f_{EM};S/E", 20, 0., 1.);
  auto* profileC = new TProfile(Form("prof_C_%s_%s", SafeName(structure.label).Data(),
                                     EnergyLabel(sample.energyGeV).Data()), ";truth f_{EM};C/E", 20, 0., 1.);
  profileS->SetDirectory(nullptr);
  profileC->SetDirectory(nullptr);
  profileS->SetMarkerColor(kBlue + 1);
  profileS->SetLineColor(kBlue + 1);
  profileS->SetMarkerStyle(20);
  profileC->SetMarkerColor(kRed + 1);
  profileC->SetLineColor(kRed + 1);
  profileC->SetMarkerStyle(21);

  for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
    tree->GetEntry(entry);
    const double truthFem = truthEdepEM / (1000. * sample.energyGeV);
    if (mcTruthEnergy <= 0. || truthFem < 0. || truthFem > 1.) continue;
    const double sEnergyMeV = scintReadout * calibration.scintMeVPerReadout;
    const double cEnergyMeV = cherenkovReadout * calibration.cherenkovMeVPerReadout;
    profileS->Fill(truthFem, sEnergyMeV / mcTruthEnergy);
    profileC->Fill(truthFem, cEnergyMeV / mcTruthEnergy);
    result.entries++;
  }

  auto* fitS = MakeHoverEFit(Form("fit_S_%s_%s", SafeName(structure.label).Data(),
                                  EnergyLabel(sample.energyGeV).Data()),
                             result.fitFemMin, result.fitFemMax);
  auto* fitC = MakeHoverEFit(Form("fit_C_%s_%s", SafeName(structure.label).Data(),
                                  EnergyLabel(sample.energyGeV).Data()),
                             result.fitFemMin, result.fitFemMax);
  fitS->SetLineColor(kBlue + 1);
  fitC->SetLineColor(kRed + 1);
  profileS->Fit(fitS, "Q0R");
  profileC->Fit(fitC, "Q0R");

  result.hoverES = fitS->GetParameter(0);
  result.hoverESErr = fitS->GetParError(0);
  result.hoverEC = fitC->GetParameter(0);
  result.hoverECErr = fitC->GetParError(0);
  result.chi2S = fitS->GetChisquare();
  result.ndfS = fitS->GetNDF();
  result.chi2C = fitC->GetChisquare();
  result.ndfC = fitC->GetNDF();

  auto* canvas = new TCanvas(Form("c_hovere_%s_%s", SafeName(structure.label).Data(),
                                  EnergyLabel(sample.energyGeV).Data()), "h/e fit", 850, 700);
  canvas->SetGrid();
  const double profileYMax = std::max({
      1.35,
      profileS->GetMaximum(),
      profileC->GetMaximum(),
      fitS->Eval(0.),
      fitS->Eval(1.),
      fitC->Eval(0.),
      fitC->Eval(1.)
  });
  profileS->SetMinimum(0.);
  profileS->SetMaximum(1.15 * profileYMax);
  profileS->Draw();
  profileC->Draw("same");
  fitS->Draw("same");
  fitC->Draw("same");

  auto* legend = new TLegend(0.12, 0.70, 0.52, 0.88);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(profileS, Form("S: h/e = %.3f #pm %.3f", result.hoverES, result.hoverESErr), "lep");
  legend->AddEntry(profileC, Form("C: h/e = %.3f #pm %.3f", result.hoverEC, result.hoverECErr), "lep");
  legend->AddEntry((TObject*)nullptr, Form("Fit %.2f < f_{EM} < %.2f", result.fitFemMin, result.fitFemMax), "");
  legend->Draw();
  auto* label = new TLatex();
  label->SetNDC();
  label->SetTextFont(42);
  label->SetTextSize(0.04);
  label->DrawLatex(0.58, 0.84, Form("%s, #pi^{-} %s", structure.label.Data(),
                                     EnergyLabel(sample.energyGeV).Data()));
  canvas->SaveAs(Form("%s/hovere_profile_%s.pdf", PlotDir(structure).Data(),
                      EnergyLabel(sample.energyGeV).Data()));
  canvas->SaveAs(Form("%s/hovere_profile_%s.png", PlotDir(structure).Data(),
                      EnergyLabel(sample.energyGeV).Data()));

  delete label;
  delete legend;
  delete canvas;
  delete fitC;
  delete fitS;
  delete profileC;
  delete profileS;
  return result;
}

HoverEFitFunction FitHoverE(const StructureInfo& structure, const Calibration& calibration)
{
  std::vector<HoverEResult> results;
  results.reserve(kNEnergy);
  for (const auto& sample : structure.pionSamples) {
    results.push_back(FitHoverEOneEnergy(structure, sample, calibration));
  }

  std::ofstream csv(Form("%s/pion_hovere_fit.csv", TableDir(structure).Data()));
  csv << "energy_GeV,entries,fit_fem_min,fit_fem_max,"
      << "h_over_e_S,h_over_e_S_err,chi2_S,ndf_S,"
      << "h_over_e_C,h_over_e_C_err,chi2_C,ndf_C\n";
  for (const auto& result : results) {
    csv << result.energyGeV << "," << result.entries << ","
        << result.fitFemMin << "," << result.fitFemMax << ","
        << result.hoverES << "," << result.hoverESErr << ","
        << result.chi2S << "," << result.ndfS << ","
        << result.hoverEC << "," << result.hoverECErr << ","
        << result.chi2C << "," << result.ndfC << "\n";
  }
  csv.close();

  double energy[kNEnergy] = {0.};
  double zero[kNEnergy] = {0.};
  double hoverES[kNEnergy] = {0.};
  double hoverESErr[kNEnergy] = {0.};
  double hoverEC[kNEnergy] = {0.};
  double hoverECErr[kNEnergy] = {0.};
  for (int i = 0; i < kNEnergy; ++i) {
    energy[i] = results[i].energyGeV;
    hoverES[i] = results[i].hoverES;
    hoverESErr[i] = results[i].hoverESErr;
    hoverEC[i] = results[i].hoverEC;
    hoverECErr[i] = results[i].hoverECErr;
  }

  auto* graphS = new TGraphErrors(kNEnergy, energy, hoverES, zero, hoverESErr);
  auto* graphC = new TGraphErrors(kNEnergy, energy, hoverEC, zero, hoverECErr);
  graphS->SetMarkerStyle(20);
  graphS->SetMarkerColor(kBlue + 1);
  graphS->SetLineColor(kBlue + 1);
  graphC->SetMarkerStyle(21);
  graphC->SetMarkerColor(kRed + 1);
  graphC->SetLineColor(kRed + 1);

  auto* fitS = new TF1(Form("fit_hovere_S_%s", SafeName(structure.label).Data()), "[0]+[1]*x", 0., 55.);
  auto* fitC = new TF1(Form("fit_hovere_C_%s", SafeName(structure.label).Data()), "[0]+[1]*x", 0., 55.);
  fitS->SetLineColor(kBlue + 1);
  fitS->SetLineStyle(2);
  fitC->SetLineColor(kRed + 1);
  fitC->SetLineStyle(2);
  graphS->Fit(fitS, "Q0");
  graphC->Fit(fitC, "Q0");

  HoverEFitFunction function;
  function.sIntercept = fitS->GetParameter(0);
  function.sSlope = fitS->GetParameter(1);
  function.cIntercept = fitC->GetParameter(0);
  function.cSlope = fitC->GetParameter(1);

  auto* canvas = new TCanvas(Form("c_hovere_summary_%s", SafeName(structure.label).Data()),
                             "h/e summary", 900, 700);
  canvas->SetGrid();
  auto* frame = new TH1D(Form("frame_hovere_summary_%s", SafeName(structure.label).Data()),
                         "", 100, 0., 55.);
  frame->SetDirectory(nullptr);
  double maxHoverE = 0.;
  for (int i = 0; i < kNEnergy; ++i) {
    maxHoverE = std::max(maxHoverE, hoverES[i] + hoverESErr[i]);
    maxHoverE = std::max(maxHoverE, hoverEC[i] + hoverECErr[i]);
  }
  maxHoverE = std::max(maxHoverE, fitS->Eval(0.));
  maxHoverE = std::max(maxHoverE, fitS->Eval(55.));
  maxHoverE = std::max(maxHoverE, fitC->Eval(0.));
  maxHoverE = std::max(maxHoverE, fitC->Eval(55.));
  frame->SetMinimum(0.);
  frame->SetMaximum(1.15 * std::max(1.2, maxHoverE));
  frame->GetXaxis()->SetTitle("Beam energy [GeV]");
  frame->GetYaxis()->SetTitle("h/e");
  frame->Draw();
  graphS->Draw("P same");
  graphC->Draw("P same");
  fitS->Draw("same");
  fitC->Draw("same");
  auto* legend = new TLegend(0.56, 0.72, 0.88, 0.88);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(graphS, "S channel", "lep");
  legend->AddEntry(graphC, "C channel", "lep");
  legend->AddEntry(fitS, Form("S: %.4f %+.5fE", function.sIntercept, function.sSlope), "l");
  legend->AddEntry(fitC, Form("C: %.4f %+.5fE", function.cIntercept, function.cSlope), "l");
  legend->Draw();
  canvas->SaveAs(Form("%s/pion_hovere_summary.pdf", PlotDir(structure).Data()));
  canvas->SaveAs(Form("%s/pion_hovere_summary.png", PlotDir(structure).Data()));

  std::ofstream fitCsv(Form("%s/pion_hovere_energy_fit.csv", TableDir(structure).Data()));
  fitCsv << "channel,intercept,intercept_err,slope_per_GeV,slope_err_per_GeV,chi2,ndf,function\n";
  fitCsv << "S," << function.sIntercept << "," << fitS->GetParError(0) << ","
         << function.sSlope << "," << fitS->GetParError(1) << ","
         << fitS->GetChisquare() << "," << fitS->GetNDF() << ","
         << "\"h_over_e(E_GeV)=" << function.sIntercept << "+(" << function.sSlope << ")*E_GeV\"\n";
  fitCsv << "C," << function.cIntercept << "," << fitC->GetParError(0) << ","
         << function.cSlope << "," << fitC->GetParError(1) << ","
         << fitC->GetChisquare() << "," << fitC->GetNDF() << ","
         << "\"h_over_e(E_GeV)=" << function.cIntercept << "+(" << function.cSlope << ")*E_GeV\"\n";
  fitCsv.close();

  delete legend;
  delete frame;
  delete canvas;
  delete fitC;
  delete fitS;
  delete graphC;
  delete graphS;
  return function;
}

double HoverES(const HoverEFitFunction& function, double energyGeV)
{
  return function.sIntercept + function.sSlope * energyGeV;
}

double HoverEC(const HoverEFitFunction& function, double energyGeV)
{
  return function.cIntercept + function.cSlope * energyGeV;
}

double DualReadoutChi(double hoverES, double hoverEC)
{
  return (1. - hoverES) / (1. - hoverEC);
}

TString OutputNtupleName(const StructureInfo& structure, const SampleInfo& sample)
{
  const std::filesystem::path inputPath(sample.fileName.Data());
  TString outputName = inputPath.filename().string();
  outputName.ReplaceAll(".root", "_dualreadout.root");
  return Form("%s/%s", NtupleDir(structure).Data(), outputName.Data());
}

ReconstructionResults ProcessDualReadoutSample(const StructureInfo& structure, const SampleInfo& sample,
                                               const Calibration& calibration,
                                               const HoverEFitFunction& hoverEFit)
{
  std::unique_ptr<TFile> input(TFile::Open(sample.fileName, "READ"));
  if (!input || input->IsZombie()) {
    std::cerr << "ERROR: cannot open " << sample.fileName << std::endl;
    return {};
  }

  auto* inputTree = dynamic_cast<TTree*>(input->Get("eventTree"));
  if (!inputTree) {
    std::cerr << "ERROR: cannot find eventTree in " << sample.fileName << std::endl;
    return {};
  }

  int scintReadout = 0;
  int cherenkovReadout = 0;
  double truthEdepEM = 0.;
  inputTree->SetBranchAddress("counter_Scintillation_ScintLayer", &scintReadout);
  inputTree->SetBranchAddress("counter_Cerenkov_CherenkovLayer", &cherenkovReadout);
  inputTree->SetBranchAddress("truthEdepEM", &truthEdepEM);

  const double hoverES = HoverES(hoverEFit, sample.energyGeV);
  const double hoverEC = HoverEC(hoverEFit, sample.energyGeV);
  const double chi = DualReadoutChi(hoverES, hoverEC);

  gSystem->mkdir(NtupleDir(structure), true);
  std::unique_ptr<TFile> output(TFile::Open(OutputNtupleName(structure, sample), "RECREATE"));
  if (!output || output->IsZombie()) {
    std::cerr << "ERROR: cannot create " << OutputNtupleName(structure, sample) << std::endl;
    return {};
  }

  const char* vectorBranches[] = {
      "Edep_Layer",
      "Nph_Cherenkov_Layer",
      "Nph_Scint_Layer",
      "vecCellID",
      "vecEdep",
      "vecNChren",
      "vecNScint"
  };
  for (const auto* branch : vectorBranches) inputTree->SetBranchStatus(branch, 0);

  output->cd();
  auto* outputTree = inputTree->CloneTree(0);
  outputTree->SetName("eventTree");
  outputTree->SetTitle("eventTree with dual-readout energy");

  double scintCalibEnergyMeV = 0.;
  double cherenkovCalibEnergyMeV = 0.;
  double dualReadoutEnergyMeV = 0.;
  double dualReadoutEnergyGeV = 0.;
  double hoverESBranch = hoverES;
  double hoverECBranch = hoverEC;
  double chiBranch = chi;
  double scintCalibrationBranch = calibration.scintMeVPerReadout;
  double cherenkovCalibrationBranch = calibration.cherenkovMeVPerReadout;

  outputTree->Branch("scintCalibEnergyMeV", &scintCalibEnergyMeV, "scintCalibEnergyMeV/D");
  outputTree->Branch("cherenkovCalibEnergyMeV", &cherenkovCalibEnergyMeV, "cherenkovCalibEnergyMeV/D");
  outputTree->Branch("dualReadoutEnergyMeV", &dualReadoutEnergyMeV, "dualReadoutEnergyMeV/D");
  outputTree->Branch("dualReadoutEnergyGeV", &dualReadoutEnergyGeV, "dualReadoutEnergyGeV/D");
  outputTree->Branch("hoverES", &hoverESBranch, "hoverES/D");
  outputTree->Branch("hoverEC", &hoverECBranch, "hoverEC/D");
  outputTree->Branch("dualReadoutChi", &chiBranch, "dualReadoutChi/D");
  outputTree->Branch("scintMeVPerReadout", &scintCalibrationBranch, "scintMeVPerReadout/D");
  outputTree->Branch("cherenkovMeVPerReadout", &cherenkovCalibrationBranch, "cherenkovMeVPerReadout/D");

  std::vector<double> dualReadoutEnergyValues;
  std::vector<double> scintEnergyValues;
  std::vector<double> cherenkovEnergyValues;
  std::vector<double> femValues;
  dualReadoutEnergyValues.reserve(inputTree->GetEntries());
  scintEnergyValues.reserve(inputTree->GetEntries());
  cherenkovEnergyValues.reserve(inputTree->GetEntries());
  femValues.reserve(inputTree->GetEntries());
  for (Long64_t entry = 0; entry < inputTree->GetEntries(); ++entry) {
    inputTree->GetEntry(entry);
    scintCalibEnergyMeV = scintReadout * calibration.scintMeVPerReadout;
    cherenkovCalibEnergyMeV = cherenkovReadout * calibration.cherenkovMeVPerReadout;
    dualReadoutEnergyMeV = (scintCalibEnergyMeV - chi * cherenkovCalibEnergyMeV) / (1. - chi);
    dualReadoutEnergyGeV = dualReadoutEnergyMeV / 1000.;
    scintEnergyValues.push_back(scintCalibEnergyMeV / 1000.);
    cherenkovEnergyValues.push_back(cherenkovCalibEnergyMeV / 1000.);
    dualReadoutEnergyValues.push_back(dualReadoutEnergyGeV);
    femValues.push_back(truthEdepEM / (1000. * sample.energyGeV));
    outputTree->Fill();
  }

  outputTree->Write();
  output->Close();
  for (const auto* branch : vectorBranches) inputTree->SetBranchStatus(branch, 1);

  ReconstructionResults results;
  results.scint = ComputeResolution(scintEnergyValues, sample.energyGeV,
                                    Form("h_scint_%s_%s", SafeName(structure.label).Data(),
                                         EnergyLabel(sample.energyGeV).Data()));
  results.cherenkov = ComputeResolution(cherenkovEnergyValues, sample.energyGeV,
                                        Form("h_cherenkov_%s_%s", SafeName(structure.label).Data(),
                                             EnergyLabel(sample.energyGeV).Data()));
  results.dualReadout = ComputeResolution(dualReadoutEnergyValues, sample.energyGeV,
                                          Form("h_dualreadout_%s_%s", SafeName(structure.label).Data(),
                                               EnergyLabel(sample.energyGeV).Data()));
  results.meanSGeV = Mean(scintEnergyValues);
  results.meanCGeV = Mean(cherenkovEnergyValues);
  results.sigmaSGeV = Rms(scintEnergyValues, results.meanSGeV);
  results.sigmaCGeV = Rms(cherenkovEnergyValues, results.meanCGeV);
  results.covSCGeV2 = Covariance(scintEnergyValues, cherenkovEnergyValues, results.meanSGeV, results.meanCGeV);
  results.rhoSC = Correlation(results.covSCGeV2, results.sigmaSGeV, results.sigmaCGeV);
  results.chi = chi;
  results.femMean = Mean(femValues);
  results.femRms = Rms(femValues, results.femMean);
  const double refVariance = (results.sigmaSGeV * results.sigmaSGeV
                           + chi * chi * results.sigmaCGeV * results.sigmaCGeV
                           - 2. * chi * results.covSCGeV2)
                           / ((1. - chi) * (1. - chi));
  results.sigmaDRRefGeV = std::sqrt(std::max(0., refVariance));
  results.resolutionDRRefPct = results.dualReadout.meanGeV > 0.
                             ? 100. * results.sigmaDRRefGeV / results.dualReadout.meanGeV : 0.;
  results.resolutionDRRefErrPct = results.resolutionDRRefPct > 0.
                                ? results.resolutionDRRefPct / std::sqrt(2. * results.dualReadout.entries) : 0.;

  DrawPionSCScatter(structure, sample, scintEnergyValues, cherenkovEnergyValues,
                    results.covSCGeV2, results.rhoSC);
  return results;
}

void BuildDualReadout(const StructureInfo& structure, const Calibration& calibration,
                      const HoverEFitFunction& hoverEFit)
{
  std::vector<ReconstructionResults> results;
  results.reserve(kNEnergy);
  for (const auto& sample : structure.pionSamples) {
    auto result = ProcessDualReadoutSample(structure, sample, calibration, hoverEFit);
    results.push_back(result);
    std::cout << Form("  %-5s S %.4f GeV  C %.4f GeV  DR %.4f GeV  DR res %.3f%%",
                      EnergyLabel(sample.energyGeV).Data(),
                      result.scint.mpvGeV,
                      result.cherenkov.mpvGeV,
                      result.dualReadout.mpvGeV,
                      result.dualReadout.resolutionPct) << std::endl;
  }

  std::ofstream csv(Form("%s/pion_dualreadout_resolution.csv", TableDir(structure).Data()));
  csv << "energy_GeV,entries,h_over_e_S,h_over_e_C,chi,"
      << "mean_S_GeV,mpv_S_GeV,sigma68_S_GeV,resolution_S_percent,"
      << "mean_C_GeV,mpv_C_GeV,sigma68_C_GeV,resolution_C_percent,"
      << "mean_DR_GeV,mpv_DR_GeV,sigma68_DR_GeV,resolution_DR_percent,resolution_DR_error_percent,"
      << "sigmaS_RMS_GeV,sigmaC_RMS_GeV,cov_SC_GeV2,rho_SC,"
      << "sigma_DR_ref_GeV,resolution_DR_ref_percent,resolution_DR_ref_error_percent,"
      << "fEM_mean,fEM_RMS,"
      << "mpv_over_beam_S,mpv_over_beam_C,mpv_over_beam_DR\n";
  for (const auto& result : results) {
    const double energyGeV = result.dualReadout.energyGeV;
    const double hoverES = HoverES(hoverEFit, energyGeV);
    const double hoverEC = HoverEC(hoverEFit, energyGeV);
    const double chi = DualReadoutChi(hoverES, hoverEC);
    csv << energyGeV << "," << result.dualReadout.entries << ","
        << hoverES << "," << hoverEC << "," << chi << ","
        << result.scint.meanGeV << "," << result.scint.mpvGeV << "," << result.scint.sigma68GeV << ","
        << result.scint.resolutionPct << ","
        << result.cherenkov.meanGeV << "," << result.cherenkov.mpvGeV << "," << result.cherenkov.sigma68GeV << ","
        << result.cherenkov.resolutionPct << ","
        << result.dualReadout.meanGeV << "," << result.dualReadout.mpvGeV << "," << result.dualReadout.sigma68GeV << ","
        << result.dualReadout.resolutionPct << "," << result.dualReadout.resolutionErrPct << ","
        << result.sigmaSGeV << "," << result.sigmaCGeV << ","
        << result.covSCGeV2 << "," << result.rhoSC << ","
        << result.sigmaDRRefGeV << "," << result.resolutionDRRefPct << ","
        << result.resolutionDRRefErrPct << ","
        << result.femMean << "," << result.femRms << ","
        << result.scint.mpvGeV / energyGeV << ","
        << result.cherenkov.mpvGeV / energyGeV << ","
        << result.dualReadout.mpvGeV / energyGeV << "\n";
  }
  csv.close();

  std::ofstream covCsv(Form("%s/pion_sc_covariance.csv", TableDir(structure).Data()));
  covCsv << "energy_GeV,entries,mean_S_GeV,mean_C_GeV,sigmaS_RMS_GeV,sigmaC_RMS_GeV,"
         << "cov_SC_GeV2,rho_SC,chi,sigma_DR_ref_GeV,resolution_DR_ref_percent\n";
  for (const auto& result : results) {
    const double energyGeV = result.dualReadout.energyGeV;
    const double chi = DualReadoutChi(HoverES(hoverEFit, energyGeV), HoverEC(hoverEFit, energyGeV));
    covCsv << energyGeV << "," << result.dualReadout.entries << ","
           << result.meanSGeV << "," << result.meanCGeV << ","
           << result.sigmaSGeV << "," << result.sigmaCGeV << ","
           << result.covSCGeV2 << "," << result.rhoSC << ","
           << chi << "," << result.sigmaDRRefGeV << "," << result.resolutionDRRefPct << "\n";
  }
  covCsv.close();

  std::ofstream femCsv(Form("%s/pion_fem_summary.csv", TableDir(structure).Data()));
  femCsv << "energy_GeV,entries,fEM_mean,fEM_RMS,definition\n";
  for (const auto& result : results) {
    femCsv << result.dualReadout.energyGeV << "," << result.dualReadout.entries << ","
           << result.femMean << "," << result.femRms << ","
           << "truthEdepEM/(1000*Ebeam_GeV)\n";
  }
  femCsv.close();

  double energy[kNEnergy] = {0.};
  double zero[kNEnergy] = {0.};
  double resolutionS[kNEnergy] = {0.};
  double resolutionC[kNEnergy] = {0.};
  double resolutionDR[kNEnergy] = {0.};
  double resolutionDRRef[kNEnergy] = {0.};
  double resolutionSErr[kNEnergy] = {0.};
  double resolutionCErr[kNEnergy] = {0.};
  double resolutionDRErr[kNEnergy] = {0.};
  double resolutionDRRefErr[kNEnergy] = {0.};
  double mpvS[kNEnergy] = {0.};
  double mpvC[kNEnergy] = {0.};
  double mpvDR[kNEnergy] = {0.};
  double mpvSErr[kNEnergy] = {0.};
  double mpvCErr[kNEnergy] = {0.};
  double mpvDRErr[kNEnergy] = {0.};
  double ratioS[kNEnergy] = {0.};
  double ratioC[kNEnergy] = {0.};
  double ratioDR[kNEnergy] = {0.};
  double ratioSErr[kNEnergy] = {0.};
  double ratioCErr[kNEnergy] = {0.};
  double ratioDRErr[kNEnergy] = {0.};
  double femMean[kNEnergy] = {0.};
  double femRms[kNEnergy] = {0.};
  double drRefSterm[kNEnergy] = {0.};
  double drRefCterm[kNEnergy] = {0.};
  double drRefCovTerm[kNEnergy] = {0.};
  double drRefStermErr[kNEnergy] = {0.};
  double drRefCtermErr[kNEnergy] = {0.};
  double drRefCovTermErr[kNEnergy] = {0.};
  double drRefCovVarianceTerm[kNEnergy] = {0.};
  for (int i = 0; i < kNEnergy; ++i) {
    energy[i] = results[i].dualReadout.energyGeV;
    resolutionS[i] = results[i].scint.resolutionPct;
    resolutionC[i] = results[i].cherenkov.resolutionPct;
    resolutionDR[i] = results[i].dualReadout.resolutionPct;
    resolutionDRRef[i] = results[i].resolutionDRRefPct;
    resolutionSErr[i] = results[i].scint.resolutionErrPct;
    resolutionCErr[i] = results[i].cherenkov.resolutionErrPct;
    resolutionDRErr[i] = results[i].dualReadout.resolutionErrPct;
    resolutionDRRefErr[i] = results[i].resolutionDRRefErrPct;
    mpvS[i] = results[i].scint.mpvGeV;
    mpvC[i] = results[i].cherenkov.mpvGeV;
    mpvDR[i] = results[i].dualReadout.mpvGeV;
    mpvSErr[i] = results[i].scint.sigma68GeV / std::sqrt(double(std::max(1, results[i].scint.entries)));
    mpvCErr[i] = results[i].cherenkov.sigma68GeV / std::sqrt(double(std::max(1, results[i].cherenkov.entries)));
    mpvDRErr[i] = results[i].dualReadout.sigma68GeV / std::sqrt(double(std::max(1, results[i].dualReadout.entries)));
    ratioS[i] = mpvS[i] / energy[i];
    ratioC[i] = mpvC[i] / energy[i];
    ratioDR[i] = mpvDR[i] / energy[i];
    ratioSErr[i] = mpvSErr[i] / energy[i];
    ratioCErr[i] = mpvCErr[i] / energy[i];
    ratioDRErr[i] = mpvDRErr[i] / energy[i];
    femMean[i] = results[i].femMean;
    femRms[i] = results[i].femRms;

    const double oneMinusChi = 1. - results[i].chi;
    const double absDenom = std::abs(oneMinusChi);
    const double meanDR = results[i].dualReadout.meanGeV;
    const double errDenom = std::sqrt(2. * std::max(1, results[i].dualReadout.entries));
    if (absDenom > 0. && meanDR > 0.) {
      const double sTermGeV = results[i].sigmaSGeV / absDenom;
      const double cTermGeV = std::abs(results[i].chi) * results[i].sigmaCGeV / absDenom;
      drRefCovVarianceTerm[i] = 2. * results[i].chi * results[i].covSCGeV2
                               / (oneMinusChi * oneMinusChi);
      const double covTermGeV = std::sqrt(std::max(0., drRefCovVarianceTerm[i]));
      drRefSterm[i] = 100. * sTermGeV / meanDR;
      drRefCterm[i] = 100. * cTermGeV / meanDR;
      drRefCovTerm[i] = 100. * covTermGeV / meanDR;
      drRefStermErr[i] = drRefSterm[i] / errDenom;
      drRefCtermErr[i] = drRefCterm[i] / errDenom;
      drRefCovTermErr[i] = drRefCovTerm[i] / errDenom;
    }
  }

  auto* canvas = new TCanvas(Form("c_dualreadout_resolution_%s", SafeName(structure.label).Data()),
                             "dual-readout resolution", 900, 700);
  canvas->SetGrid();
  auto* frame = new TH1D(Form("frame_dualreadout_resolution_%s", SafeName(structure.label).Data()),
                         "", 100, 0., 55.);
  frame->SetDirectory(nullptr);
  frame->SetMinimum(0.);
  double maxResolution = 0.;
  for (int i = 0; i < kNEnergy; ++i) {
    maxResolution = std::max(maxResolution, resolutionS[i] + resolutionSErr[i]);
    maxResolution = std::max(maxResolution, resolutionC[i] + resolutionCErr[i]);
    maxResolution = std::max(maxResolution, resolutionDR[i] + resolutionDRErr[i]);
  }
  frame->SetMaximum(std::max(30., 1.20 * maxResolution));
  frame->GetXaxis()->SetTitle("Beam energy [GeV]");
  frame->GetYaxis()->SetTitle("#sigma_{68} / MPV [%]");
  frame->Draw();

  auto* graphS = new TGraphErrors(kNEnergy, energy, resolutionS, zero, resolutionSErr);
  graphS->SetMarkerStyle(20);
  graphS->SetMarkerSize(1.1);
  graphS->SetMarkerColor(kRed + 1);
  graphS->SetLineColor(kRed + 1);
  graphS->Draw("P same");

  auto* graphC = new TGraphErrors(kNEnergy, energy, resolutionC, zero, resolutionCErr);
  graphC->SetMarkerStyle(21);
  graphC->SetMarkerSize(1.1);
  graphC->SetMarkerColor(kGreen + 2);
  graphC->SetLineColor(kGreen + 2);
  graphC->Draw("P same");

  auto* graphDR = new TGraphErrors(kNEnergy, energy, resolutionDR, zero, resolutionDRErr);
  graphDR->SetMarkerStyle(22);
  graphDR->SetMarkerSize(1.1);
  graphDR->SetMarkerColor(kBlue + 1);
  graphDR->SetLineColor(kBlue + 1);
  graphDR->Draw("P same");

  auto* fitSRes = new TF1(Form("fit_scint_resolution_%s", SafeName(structure.label).Data()),
                          "sqrt([0]*[0]/x + [1]*[1])", 1., 55.);
  fitSRes->SetParNames("a", "b");
  fitSRes->SetParameters(40., 2.);
  graphS->Fit(fitSRes, "Q0");
  fitSRes->SetLineColor(kRed + 1);
  fitSRes->SetLineWidth(2);
  fitSRes->Draw("same");

  auto* fitCRes = new TF1(Form("fit_cherenkov_resolution_%s", SafeName(structure.label).Data()),
                          "sqrt([0]*[0]/x + [1]*[1])", 1., 55.);
  fitCRes->SetParNames("a", "b");
  fitCRes->SetParameters(40., 2.);
  graphC->Fit(fitCRes, "Q0");
  fitCRes->SetLineColor(kGreen + 2);
  fitCRes->SetLineWidth(2);
  fitCRes->Draw("same");

  auto* fitDRRes = new TF1(Form("fit_dualreadout_resolution_%s", SafeName(structure.label).Data()),
                           "sqrt([0]*[0]/x + [1]*[1])", 1., 55.);
  fitDRRes->SetParNames("a", "b");
  fitDRRes->SetParameters(40., 2.);
  graphDR->Fit(fitDRRes, "Q0");
  fitDRRes->SetLineColor(kBlue + 1);
  fitDRRes->SetLineWidth(2);
  fitDRRes->Draw("same");

  auto* legend = new TLegend(0.32, 0.58, 0.88, 0.88);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(graphS, "S only", "lep");
  legend->AddEntry(fitSRes, Form("#sigma/MPV = %.2f%%/#sqrt{E} #oplus %.2f%%",
                                 fitSRes->GetParameter(0), fitSRes->GetParameter(1)), "l");
  legend->AddEntry(graphC, "C only", "lep");
  legend->AddEntry(fitCRes, Form("#sigma/MPV = %.2f%%/#sqrt{E} #oplus %.2f%%",
                                 fitCRes->GetParameter(0), fitCRes->GetParameter(1)), "l");
  legend->AddEntry(graphDR, "Dual readout", "lep");
  legend->AddEntry(fitDRRes, Form("#sigma/MPV = %.2f%%/#sqrt{E} #oplus %.2f%%",
                                  fitDRRes->GetParameter(0), fitDRRes->GetParameter(1)), "l");
  legend->Draw();

  canvas->SaveAs(Form("%s/pion_dualreadout_energy_resolution.pdf", PlotDir(structure).Data()));
  canvas->SaveAs(Form("%s/pion_dualreadout_energy_resolution.png", PlotDir(structure).Data()));

  auto* decompCanvas = new TCanvas(Form("c_dualreadout_covariance_decomposition_%s",
                                        SafeName(structure.label).Data()),
                                   "dual-readout covariance decomposition", 1000, 760);
  decompCanvas->SetGrid();
  auto* decompFrame = new TH1D(Form("frame_dualreadout_covariance_decomposition_%s",
                                    SafeName(structure.label).Data()),
                               "", 100, 0., 55.);
  decompFrame->SetDirectory(nullptr);
  decompFrame->SetMinimum(0.);
  double maxDecomp = 0.;
  for (int i = 0; i < kNEnergy; ++i) {
    maxDecomp = std::max(maxDecomp, resolutionDRRef[i] + resolutionDRRefErr[i]);
    maxDecomp = std::max(maxDecomp, drRefSterm[i] + drRefStermErr[i]);
    maxDecomp = std::max(maxDecomp, drRefCterm[i] + drRefCtermErr[i]);
    maxDecomp = std::max(maxDecomp, drRefCovTerm[i] + drRefCovTermErr[i]);
    maxDecomp = std::max(maxDecomp, resolutionDR[i] + resolutionDRErr[i]);
  }
  decompFrame->SetMaximum(std::max(30., 1.25 * maxDecomp));
  decompFrame->GetXaxis()->SetTitle("Beam energy [GeV]");
  decompFrame->GetYaxis()->SetTitle("Resolution component / #LT E_{DR}#GT [%]");
  decompFrame->Draw();

  auto* graphDRRef = new TGraphErrors(kNEnergy, energy, resolutionDRRef, zero, resolutionDRRefErr);
  graphDRRef->SetMarkerStyle(23);
  graphDRRef->SetMarkerSize(1.1);
  graphDRRef->SetMarkerColor(kMagenta + 1);
  graphDRRef->SetLineColor(kMagenta + 1);
  graphDRRef->Draw("P same");

  auto* graphSterm = new TGraphErrors(kNEnergy, energy, drRefSterm, zero, drRefStermErr);
  graphSterm->SetMarkerStyle(24);
  graphSterm->SetMarkerSize(1.1);
  graphSterm->SetMarkerColor(kRed + 1);
  graphSterm->SetLineColor(kRed + 1);
  graphSterm->Draw("P same");

  auto* graphCterm = new TGraphErrors(kNEnergy, energy, drRefCterm, zero, drRefCtermErr);
  graphCterm->SetMarkerStyle(25);
  graphCterm->SetMarkerSize(1.1);
  graphCterm->SetMarkerColor(kGreen + 2);
  graphCterm->SetLineColor(kGreen + 2);
  graphCterm->Draw("P same");

  auto* graphCovTerm = new TGraphErrors(kNEnergy, energy, drRefCovTerm, zero, drRefCovTermErr);
  graphCovTerm->SetMarkerStyle(26);
  graphCovTerm->SetMarkerSize(1.1);
  graphCovTerm->SetMarkerColor(kOrange + 7);
  graphCovTerm->SetLineColor(kOrange + 7);
  graphCovTerm->Draw("P same");

  auto* graphDRMeasured = new TGraphErrors(kNEnergy, energy, resolutionDR, zero, resolutionDRErr);
  graphDRMeasured->SetMarkerStyle(22);
  graphDRMeasured->SetMarkerSize(1.1);
  graphDRMeasured->SetMarkerColor(kBlue + 1);
  graphDRMeasured->SetLineColor(kBlue + 1);
  graphDRMeasured->Draw("P same");

  auto makeResolutionFit = [&](const char* name, Color_t color, Style_t style, double startA) {
    auto* fit = new TF1(Form("%s_%s", name, SafeName(structure.label).Data()),
                        "sqrt([0]*[0]/x + [1]*[1])", 1., 55.);
    fit->SetParNames("a", "b");
    fit->SetParameters(startA, 2.);
    fit->SetLineColor(color);
    fit->SetLineWidth(2);
    fit->SetLineStyle(style);
    return fit;
  };

  auto* fitDRRefRes = makeResolutionFit("fit_dualreadout_ref_resolution", kMagenta + 1, 7, 40.);
  graphDRRef->Fit(fitDRRefRes, "Q0");
  fitDRRefRes->Draw("same");
  auto* fitStermRes = makeResolutionFit("fit_dualreadout_s_term_resolution", kRed + 1, 2, 50.);
  graphSterm->Fit(fitStermRes, "Q0");
  fitStermRes->Draw("same");
  auto* fitCtermRes = makeResolutionFit("fit_dualreadout_c_term_resolution", kGreen + 2, 3, 50.);
  graphCterm->Fit(fitCtermRes, "Q0");
  fitCtermRes->Draw("same");
  auto* fitCovTermRes = makeResolutionFit("fit_dualreadout_cov_term_resolution", kOrange + 7, 4, 50.);
  graphCovTerm->Fit(fitCovTermRes, "Q0");
  fitCovTermRes->Draw("same");
  auto* fitDRMeasuredRes = makeResolutionFit("fit_dualreadout_measured_resolution", kBlue + 1, 1, 40.);
  graphDRMeasured->Fit(fitDRMeasuredRes, "Q0");
  fitDRMeasuredRes->Draw("same");

  auto* decompLegend = new TLegend(0.28, 0.52, 0.90, 0.89);
  decompLegend->SetBorderSize(0);
  decompLegend->SetFillStyle(0);
  decompLegend->AddEntry(graphDRRef, Form("#sigma_{DR}^{ref}, %.2f%%/#sqrt{E} #oplus %.2f%%",
                                           fitDRRefRes->GetParameter(0), fitDRRefRes->GetParameter(1)), "lep");
  decompLegend->AddEntry(graphSterm, Form("#sigma_{S}/|1-#chi|, %.2f%%/#sqrt{E} #oplus %.2f%%",
                                           fitStermRes->GetParameter(0), fitStermRes->GetParameter(1)), "lep");
  decompLegend->AddEntry(graphCterm, Form("|#chi|#sigma_{C}/|1-#chi|, %.2f%%/#sqrt{E} #oplus %.2f%%",
                                           fitCtermRes->GetParameter(0), fitCtermRes->GetParameter(1)), "lep");
  decompLegend->AddEntry(graphCovTerm, Form("#sqrt{2#chi cov(S,C)}/|1-#chi|, %.2f%%/#sqrt{E} #oplus %.2f%%",
                                             fitCovTermRes->GetParameter(0), fitCovTermRes->GetParameter(1)), "lep");
  decompLegend->AddEntry(graphDRMeasured, Form("Measured E_{DR}, %.2f%%/#sqrt{E} #oplus %.2f%%",
                                                fitDRMeasuredRes->GetParameter(0),
                                                fitDRMeasuredRes->GetParameter(1)), "lep");
  decompLegend->Draw();

  decompCanvas->SaveAs(Form("%s/pion_dualreadout_covariance_decomposition.pdf", PlotDir(structure).Data()));
  decompCanvas->SaveAs(Form("%s/pion_dualreadout_covariance_decomposition.png", PlotDir(structure).Data()));

  std::ofstream decompCsv(Form("%s/pion_dualreadout_covariance_decomposition.csv",
                               TableDir(structure).Data()));
  decompCsv << "energy_GeV,entries,chi,mean_DR_GeV,"
            << "sigmaS_over_abs1minuschi_GeV,resolution_sigmaS_over_abs1minuschi_percent,"
            << "abschi_sigmaC_over_abs1minuschi_GeV,resolution_abschi_sigmaC_over_abs1minuschi_percent,"
            << "covariance_variance_term_GeV2,covariance_cancellation_sigma_GeV,"
            << "resolution_covariance_cancellation_percent,"
            << "sigma_DR_ref_GeV,resolution_DR_ref_percent,"
            << "sigma68_DR_GeV,resolution_DR_percent\n";
  for (int i = 0; i < kNEnergy; ++i) {
    const double oneMinusChi = 1. - results[i].chi;
    const double absDenom = std::abs(oneMinusChi);
    const double meanDR = results[i].dualReadout.meanGeV;
    const double sTermGeV = absDenom > 0. ? results[i].sigmaSGeV / absDenom : 0.;
    const double cTermGeV = absDenom > 0. ? std::abs(results[i].chi) * results[i].sigmaCGeV / absDenom : 0.;
    const double covTermGeV = std::sqrt(std::max(0., drRefCovVarianceTerm[i]));
    decompCsv << energy[i] << "," << results[i].dualReadout.entries << "," << results[i].chi << ","
              << meanDR << "," << sTermGeV << "," << drRefSterm[i] << ","
              << cTermGeV << "," << drRefCterm[i] << ","
              << drRefCovVarianceTerm[i] << "," << covTermGeV << "," << drRefCovTerm[i] << ","
              << results[i].sigmaDRRefGeV << "," << resolutionDRRef[i] << ","
              << results[i].dualReadout.sigma68GeV << "," << resolutionDR[i] << "\n";
  }
  decompCsv.close();

  std::ofstream decompFitCsv(Form("%s/pion_dualreadout_covariance_decomposition_fit.csv",
                                  TableDir(structure).Data()));
  decompFitCsv << "channel,a_percent_sqrtGeV,a_err,b_percent,b_err,chi2,ndf,function\n";
  auto writeDecompFit = [&](const char* channel, const TF1* fit) {
    decompFitCsv << channel << "," << fit->GetParameter(0) << "," << fit->GetParError(0) << ","
                 << fit->GetParameter(1) << "," << fit->GetParError(1) << ","
                 << fit->GetChisquare() << "," << fit->GetNDF() << ","
                 << "\"resolution_percent(E_GeV)=sqrt((" << fit->GetParameter(0)
                 << ")^2/E_GeV+(" << fit->GetParameter(1) << ")^2)\"\n";
  };
  writeDecompFit("DR_ref", fitDRRefRes);
  writeDecompFit("sigmaS_over_abs1minuschi", fitStermRes);
  writeDecompFit("abschi_sigmaC_over_abs1minuschi", fitCtermRes);
  writeDecompFit("covariance_cancellation", fitCovTermRes);
  writeDecompFit("DR_measured", fitDRMeasuredRes);
  decompFitCsv.close();

  auto* femCanvas = new TCanvas(Form("c_pion_fem_summary_%s", SafeName(structure.label).Data()),
                                "pion fEM summary", 850, 700);
  femCanvas->SetGrid();
  auto* femFrame = new TH1D(Form("frame_pion_fem_summary_%s", SafeName(structure.label).Data()),
                            "", 100, 0., 55.);
  femFrame->SetDirectory(nullptr);
  femFrame->SetMinimum(0.);
  femFrame->SetMaximum(1.);
  femFrame->GetXaxis()->SetTitle("Beam energy [GeV]");
  femFrame->GetYaxis()->SetTitle("#LT f_{EM} #GT with RMS");
  femFrame->Draw();
  auto* graphFem = new TGraphErrors(kNEnergy, energy, femMean, zero, femRms);
  graphFem->SetMarkerStyle(20);
  graphFem->SetMarkerSize(1.1);
  graphFem->SetMarkerColor(kBlue + 1);
  graphFem->SetLineColor(kBlue + 1);
  graphFem->Draw("P same");
  auto* femLegend = new TLegend(0.16, 0.78, 0.72, 0.88);
  femLegend->SetBorderSize(0);
  femLegend->SetFillStyle(0);
  femLegend->AddEntry(graphFem, "f_{EM} = truthEdepEM / E_{beam}; error bar = RMS", "lep");
  femLegend->Draw();
  femCanvas->SaveAs(Form("%s/pion_fem_mean_rms.pdf", PlotDir(structure).Data()));
  femCanvas->SaveAs(Form("%s/pion_fem_mean_rms.png", PlotDir(structure).Data()));

  auto* linearityCanvas = new TCanvas(Form("c_energy_linearity_%s", SafeName(structure.label).Data()),
                                      "pion energy linearity", 950, 850);
  auto* topPad = new TPad(Form("pad_linearity_top_%s", SafeName(structure.label).Data()), "", 0., 0.30, 1., 1.);
  auto* bottomPad = new TPad(Form("pad_linearity_ratio_%s", SafeName(structure.label).Data()), "", 0., 0., 1., 0.30);
  topPad->SetBottomMargin(0.02);
  bottomPad->SetTopMargin(0.04);
  bottomPad->SetBottomMargin(0.32);
  topPad->Draw();
  bottomPad->Draw();

  topPad->cd();
  topPad->SetGrid();
  auto* linearityFrame = new TH1D(Form("frame_energy_linearity_%s", SafeName(structure.label).Data()),
                                  "", 100, 0., 55.);
  linearityFrame->SetDirectory(nullptr);
  linearityFrame->SetMinimum(0.);
  linearityFrame->SetMaximum(55.);
  linearityFrame->GetXaxis()->SetLabelSize(0.);
  linearityFrame->GetYaxis()->SetTitle("Reconstructed MPV [GeV]");
  linearityFrame->GetYaxis()->SetTitleSize(0.045);
  linearityFrame->GetYaxis()->SetTitleOffset(1.05);
  linearityFrame->Draw();

  auto* graphSLinearity = new TGraphErrors(kNEnergy, energy, mpvS, zero, mpvSErr);
  graphSLinearity->SetMarkerStyle(20);
  graphSLinearity->SetMarkerColor(kBlue + 1);
  graphSLinearity->SetLineColor(kBlue + 1);
  graphSLinearity->Draw("P same");
  auto* graphCLinearity = new TGraphErrors(kNEnergy, energy, mpvC, zero, mpvCErr);
  graphCLinearity->SetMarkerStyle(21);
  graphCLinearity->SetMarkerColor(kRed + 1);
  graphCLinearity->SetLineColor(kRed + 1);
  graphCLinearity->Draw("P same");
  auto* graphDRLinearity = new TGraphErrors(kNEnergy, energy, mpvDR, zero, mpvDRErr);
  graphDRLinearity->SetMarkerStyle(22);
  graphDRLinearity->SetMarkerColor(kGreen + 2);
  graphDRLinearity->SetLineColor(kGreen + 2);
  graphDRLinearity->Draw("P same");

  auto* idealLine = new TLine(0., 0., 55., 55.);
  idealLine->SetLineColor(kGray + 2);
  idealLine->SetLineStyle(2);
  idealLine->SetLineWidth(2);
  idealLine->Draw("same");

  auto* linearityLegend = new TLegend(0.16, 0.70, 0.50, 0.88);
  linearityLegend->SetBorderSize(0);
  linearityLegend->SetFillStyle(0);
  linearityLegend->AddEntry(graphSLinearity, "E_{S}", "lep");
  linearityLegend->AddEntry(graphCLinearity, "E_{C}", "lep");
  linearityLegend->AddEntry(graphDRLinearity, "E_{DR}", "lep");
  linearityLegend->AddEntry(idealLine, "MPV = E_{beam}", "l");
  linearityLegend->Draw();

  bottomPad->cd();
  bottomPad->SetGrid();
  auto* ratioFrame = new TH1D(Form("frame_energy_ratio_%s", SafeName(structure.label).Data()), "", 100, 0., 55.);
  ratioFrame->SetDirectory(nullptr);
  ratioFrame->SetMinimum(0.4);
  ratioFrame->SetMaximum(1.25);
  ratioFrame->GetXaxis()->SetTitle("Beam energy [GeV]");
  ratioFrame->GetYaxis()->SetTitle("MPV / E_{beam}");
  ratioFrame->GetXaxis()->SetTitleSize(0.11);
  ratioFrame->GetXaxis()->SetLabelSize(0.09);
  ratioFrame->GetYaxis()->SetTitleSize(0.09);
  ratioFrame->GetYaxis()->SetLabelSize(0.08);
  ratioFrame->GetYaxis()->SetTitleOffset(0.50);
  ratioFrame->GetYaxis()->SetNdivisions(505);
  ratioFrame->Draw();

  auto* graphSRatio = new TGraphErrors(kNEnergy, energy, ratioS, zero, ratioSErr);
  graphSRatio->SetMarkerStyle(20);
  graphSRatio->SetMarkerColor(kBlue + 1);
  graphSRatio->SetLineColor(kBlue + 1);
  graphSRatio->Draw("P same");
  auto* graphCRatio = new TGraphErrors(kNEnergy, energy, ratioC, zero, ratioCErr);
  graphCRatio->SetMarkerStyle(21);
  graphCRatio->SetMarkerColor(kRed + 1);
  graphCRatio->SetLineColor(kRed + 1);
  graphCRatio->Draw("P same");
  auto* graphDRRatio = new TGraphErrors(kNEnergy, energy, ratioDR, zero, ratioDRErr);
  graphDRRatio->SetMarkerStyle(22);
  graphDRRatio->SetMarkerColor(kGreen + 2);
  graphDRRatio->SetLineColor(kGreen + 2);
  graphDRRatio->Draw("P same");

  auto* unityLine = new TLine(0., 1., 55., 1.);
  unityLine->SetLineColor(kGray + 2);
  unityLine->SetLineStyle(2);
  unityLine->SetLineWidth(2);
  unityLine->Draw("same");

  linearityCanvas->SaveAs(Form("%s/pion_energy_linearity.pdf", PlotDir(structure).Data()));
  linearityCanvas->SaveAs(Form("%s/pion_energy_linearity.png", PlotDir(structure).Data()));

  std::ofstream fitCsv(Form("%s/pion_dualreadout_resolution_fit.csv", TableDir(structure).Data()));
  fitCsv << "channel,a_percent_sqrtGeV,a_err,b_percent,b_err,chi2,ndf,function\n";
  fitCsv << "S," << fitSRes->GetParameter(0) << "," << fitSRes->GetParError(0) << ","
         << fitSRes->GetParameter(1) << "," << fitSRes->GetParError(1) << ","
         << fitSRes->GetChisquare() << "," << fitSRes->GetNDF() << ","
         << "\"resolution_percent(E_GeV)=sqrt((" << fitSRes->GetParameter(0)
         << ")^2/E_GeV+(" << fitSRes->GetParameter(1) << ")^2)\"\n";
  fitCsv << "C," << fitCRes->GetParameter(0) << "," << fitCRes->GetParError(0) << ","
         << fitCRes->GetParameter(1) << "," << fitCRes->GetParError(1) << ","
         << fitCRes->GetChisquare() << "," << fitCRes->GetNDF() << ","
         << "\"resolution_percent(E_GeV)=sqrt((" << fitCRes->GetParameter(0)
         << ")^2/E_GeV+(" << fitCRes->GetParameter(1) << ")^2)\"\n";
  fitCsv << "DR," << fitDRRes->GetParameter(0) << "," << fitDRRes->GetParError(0) << ","
         << fitDRRes->GetParameter(1) << "," << fitDRRes->GetParError(1) << ","
         << fitDRRes->GetChisquare() << "," << fitDRRes->GetNDF() << ","
         << "\"resolution_percent(E_GeV)=sqrt((" << fitDRRes->GetParameter(0)
         << ")^2/E_GeV+(" << fitDRRes->GetParameter(1) << ")^2)\"\n";
  fitCsv.close();

  delete unityLine;
  delete graphDRRatio;
  delete graphCRatio;
  delete graphSRatio;
  delete ratioFrame;
  delete linearityLegend;
  delete idealLine;
  delete graphDRLinearity;
  delete graphCLinearity;
  delete graphSLinearity;
  delete linearityFrame;
  delete bottomPad;
  delete topPad;
  delete linearityCanvas;

  delete femLegend;
  delete graphFem;
  delete femFrame;
  delete femCanvas;

  delete decompLegend;
  delete fitDRMeasuredRes;
  delete fitCovTermRes;
  delete fitCtermRes;
  delete fitStermRes;
  delete fitDRRefRes;
  delete graphDRMeasured;
  delete graphCovTerm;
  delete graphCterm;
  delete graphSterm;
  delete graphDRRef;
  delete decompFrame;
  delete decompCanvas;

  delete legend;
  delete fitDRRes;
  delete fitCRes;
  delete fitSRes;
  delete graphDR;
  delete graphC;
  delete graphS;
  delete frame;
  delete canvas;
}

}  // namespace

void RunFullDualReadout()
{
  gStyle->SetOptStat(0);
  const auto structures = FindStructures();
  std::cout << "Found " << structures.size() << " complete structures." << std::endl;

  gSystem->mkdir(kOutputDir, true);
  std::ofstream summary(Form("%s/summary.csv", kOutputDir.Data()));
  summary << "structure,scint_MeV_per_readout,cherenkov_MeV_per_readout,"
          << "h_over_e_S_intercept,h_over_e_S_slope,h_over_e_C_intercept,h_over_e_C_slope\n";

  for (const auto& structure : structures) {
    std::cout << "\n=== " << structure.label << " ===" << std::endl;
    //if(structure.label!="ZnWO4_10_Quartz_5_Steel_10" && structure.label!="ZnWO4_8_Quartz_4_Steel_4" && structure.label!="ZnWO4_8_Quartz_4_Steel_8" ) continue;

    gSystem->mkdir(PlotDir(structure), true);
    gSystem->mkdir(TableDir(structure), true);
    gSystem->mkdir(NtupleDir(structure), true);

    AnalyzeSamplingFractions(structure);

    const auto calibration = CalibrateAndPlotElectrons(structure);
    std::cout << Form("  EM calibration: S %.6g MeV/count, C %.6g MeV/count",
                      calibration.scintMeVPerReadout, calibration.cherenkovMeVPerReadout) << std::endl;

    const auto hoverEFit = FitHoverE(structure, calibration);
    std::cout << Form("  h/e S(E)=%.6f%+.6fE, C(E)=%.6f%+.6fE",
                      hoverEFit.sIntercept, hoverEFit.sSlope,
                      hoverEFit.cIntercept, hoverEFit.cSlope) << std::endl;

    BuildDualReadout(structure, calibration, hoverEFit);

    summary << structure.label << ","
            << calibration.scintMeVPerReadout << "," << calibration.cherenkovMeVPerReadout << ","
            << hoverEFit.sIntercept << "," << hoverEFit.sSlope << ","
            << hoverEFit.cIntercept << "," << hoverEFit.cSlope << "\n";
  }
  summary.close();

  std::cout << "\nWrote " << kOutputDir << "/<structure>/{plots,tables,ntuples}" << std::endl;
  std::cout << "Wrote " << kOutputDir << "/summary.csv" << std::endl;
}
