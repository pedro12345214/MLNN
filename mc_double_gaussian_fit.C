using namespace RooFit;

#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TF1.h>
#include <TStyle.h>
#include <TLegend.h>
#include <iostream>
#include "ACCSEL.h"

#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooGaussian.h> 
#include <RooCBShape.h>
#include <RooAddPdf.h>
#include <RooPlot.h>
#include <RooFitResult.h>
#include <RooProduct.h>
#include <RooFormulaVar.h>
#include <RooExponential.h>
#include <RooGenericPdf.h>  
#include <RooProdPdf.h>
#include <RooCBShape.h>
#include <RooBifurGauss.h>
#include <RooCategory.h>
#include <RooSimultaneous.h>

#include <RooFitResult.h>
#include <RooFit.h>
#include <RooCmdArg.h>




struct FitParams {
    double interval_fit_min, interval_fit_max;
    double mean_init, mean_min, mean_max;
    double sigma1_init, sigma1_min, sigma1_max;
    double sigma2_init, sigma2_min, sigma2_max;
    double c1_init, c1_min, c1_max;
    double N_min, N_max;
};


void fit_mc_signal_roofit(TString particle) {
    // Abrir ficheiro ROOT, obter a árvore, nome do eixo dos xx e definições do histograma, dependendo da partícula
    TTree* tree = nullptr;
    TString path_to_file = "";
    TString file_name = "";    
    TString path_to_tree = "";
    TString X_Axis_Title = "";
    TString legend_name = "";

    // Bins do Gráfico
    double xlow = 0.0;
    double xhigh = 0.0;
    double interval_of_bins = 0.0;
    int nbins = 0;

    // Posição das Legendas
    double x_min_legend = 0.0;
    double x_max_legend = 0.0;

    if (particle == "Bd"){
        path_to_file = "/lstore/cms/lekai/Bmeson/MC/ppRef/Bd_phat5_Bfinder.root";
        ///lstore/cms/hlegoinha/Bmeson/MC_DATA/MC_ppRef_Bmeson/Bd_phat5_Bfinder.root
        file_name = "Bd_Gaussian_Fit.pdf";
        path_to_tree = "Bfinder/ntKstar";
        X_Axis_Title = "m_{J/#Psi K^{*}} [GeV/c^{2}]";
        xlow = 5.0;
        xhigh = 6.0;
        interval_of_bins = 0.0075;
        nbins = int((xhigh - xlow) / interval_of_bins);
        x_min_legend = 0.50;
        x_max_legend = 0.88;
        legend_name = "B0";
        
    }
    else if (particle == "Bu"){
        path_to_file = "/lstore/cms/hlegoinha/DATA_Sharing/Bmesons/MC_Bu.root";
        // /lstore/cms/hlegoinha/Bmeson/MC_DATA/MC_ppRef_Bmeson/Bu_phat5_Bfinder.root
        file_name = "Bu_Gaussian_Fit.pdf";
        path_to_tree = "ntKp";
        X_Axis_Title = "m_{J/#Psi K^{+}} [GeV/c^{2}]";
        xlow = 5.0;
        xhigh = 6.0;
        interval_of_bins = 0.0075;
        nbins = int((xhigh - xlow) / interval_of_bins);
        x_min_legend = 0.50;
        x_max_legend = 0.88;
        legend_name = "B^{+}";
    }
    else if (particle == "PSI2S"){
        path_to_file = "/lstore/cms/hlegoinha/X3872/MC_DATA/prompt_PSI2S_to_Jpsi_pipi_phat5_Bfinder.root";
        file_name = "PSI2S_Gaussian_Fit.pdf";
        path_to_tree ="Bfinder/ntmix";
        X_Axis_Title = "m_{J/#Psi#pi^{+}#pi^{-}} [GeV/c^{2}]";
        xlow = 3.6;
        xhigh = 4.0;
        interval_of_bins = 0.0025;
        nbins = int((xhigh - xlow) / interval_of_bins);
        x_min_legend = 0.50;
        x_max_legend = 0.88;
        legend_name = "PSI2S";
    }
    else if (particle == "X3872"){
        path_to_file = "/lstore/cms/hlegoinha/X3872/MC_DATA/prompt_X3872_to_Jpsi_Rho_phat5_Bfinder.root";
        file_name = "X3872_Gaussian_Fit.pdf";
        path_to_tree = "Bfinder/ntmix";
        X_Axis_Title = "m_{J/#Psi#rho} [GeV/c^{2}]";
        xlow = 3.6;
        xhigh = 4.0;
        interval_of_bins = 0.0025;
        nbins = int((xhigh - xlow) / interval_of_bins);
        x_min_legend = 0.14;
        x_max_legend = 0.52;
        legend_name = "X3872";
    }

    // Verificar se o Ficheiro e a Tree são encontrados
    TFile* file = TFile::Open(path_to_file);
    if (!file || file->IsZombie()) {
        std::cerr << "Ficheiro de MC não encontrado!" << std::endl;
        return;
    }

    file->GetObject(path_to_tree, tree);
    if (!tree) {
        std::cerr << "Tree não encontrada!" << std::endl;
        return;
    }

    // Definir cortes
    TString cut = "";
    if (particle == "Bu") {
        cut = Form("(%s) && (%s) && (%s) && (%s)",
                isMCsignal.Data(),
                ACCcuts_ppRef_Bu.Data(),
                SELcuts_ppRef_Bu.Data(),
                TRGmatching.Data());
    }
    else {
        cut = Form("(%s) && (%s) && (%s) && (%s)",
                isMCsignal.Data(),
                ACCcuts_ppRef.Data(),
                SELcuts_ppRef.Data(),
                TRGmatching.Data());
    }
    
    // Preencher histograma
    TH1F* h = new TH1F("h_roofit", "MC Signal;Bmass [GeV/c^{2}];Entries", nbins, xlow, xhigh);
    tree->Draw("Bmass >> h_roofit", cut, "goff");

    // Verificar se o histograma não fica vazio após os cortes (se fica, é sinal que algo está mal)
    if (h->GetEntries() == 0) {
        std::cerr << "Histograma vazio após os cortes!" << std::endl;
        delete h;
        return;
    }

    // Normalizar o histograma
    // h->Scale(1.0 / h->Integral());

    // VALORES INICIAIS E LIMITES DOS PARÂMETROS
    FitParams params;
    if (particle == "Bd"){
        params.interval_fit_min = 5.18;
        params.interval_fit_max = 5.38;

        params.mean_init = 5.28;
        params.mean_min = 5.26;
        params.mean_max = 5.3;

        params.sigma1_init = 0.014;
        params.sigma1_min = 0.001;
        params.sigma1_max = 0.07;

        params.sigma2_init = 0.034;
        params.sigma2_min = 0.001;
        params.sigma2_max = 0.09;

        params.c1_init = 0.8;
        params.c1_min  = 0.01;
        params.c1_max  = 0.99;
    
        params.N_min = 0;
        params.N_max = 90000;
    }
    else if (particle == "Bu"){
        params.interval_fit_min = 5.18;
        params.interval_fit_max = 5.42;

        params.mean_init = 5.28;
        params.mean_min = 5.26;
        params.mean_max = 5.3;

        params.sigma1_init = 0.036;
        params.sigma1_min = 0.001;
        params.sigma1_max = 0.09;

        params.sigma2_init = 0.011;
        params.sigma2_min = 0.001;
        params.sigma2_max = 0.07;

        params.c1_init = 0.52;
        params.c1_min  = 0.01;
        params.c1_max  = 0.99;   
    
        params.N_min = 0;
        params.N_max = 90000;
    }
    else if (particle == "PSI2S"){
        params.interval_fit_min = 3.64;
        params.interval_fit_max = 3.73;

        params.mean_init = 3.686;
        params.mean_min = 3.6;
        params.mean_max = 3.8;

        params.sigma1_init = 0.0048;
        params.sigma1_min = 0.001;
        params.sigma1_max = 0.09;

        params.sigma2_init = 0.0099;
        params.sigma2_min = 0.001;
        params.sigma2_max = 0.09;

        params.c1_init = 0.66;
        params.c1_min  = 0.1;
        params.c1_max  = 0.9;
    
        params.N_min = 0;
        params.N_max = 100000;
    }
    else if (particle == "X3872"){
        params.interval_fit_min = 3.8;
        params.interval_fit_max = 3.95;

        params.mean_init = 3.872;
        params.mean_min = 3.8;
        params.mean_max = 3.9;

        params.sigma1_init = 0.0056;
        params.sigma1_min = 0.001;
        params.sigma1_max = 0.08;

        params.sigma2_init = 0.0125;
        params.sigma2_min = 0.001;
        params.sigma2_max = 0.08;

        params.c1_init = 0.473;
        params.c1_min  = 0.01;
        params.c1_max  = 0.99;
    
        params.N_min = 0;
        params.N_max = 100000;
    }

    // RooFit: variável observável
    RooRealVar Bmass("Bmass", "Bmass", xlow, xhigh);
    Bmass.setRange("fitRange", params.interval_fit_min, params.interval_fit_max);
    RooDataHist data("data", "Histograma RooFit", RooArgList(Bmass), h);

    // Parâmetros do fit
    RooRealVar mean("mean", "mean", params.mean_init, params.mean_min, params.mean_max);
    RooRealVar sigma1("sigma1", "sigma1", params.sigma1_init, params.sigma1_min, params.sigma1_max);
    RooRealVar sigma2("sigma2", "sigma2", params.sigma2_init, params.sigma2_min, params.sigma2_max);
    RooRealVar c1("c1", "Fraction of G1", 0.65, 0.0, 1.0);

    RooRealVar N("N", "Total yield", h->GetEntries(), params.N_min, params.N_max);

    // Gaussianas
    RooGaussian gauss1("gauss1", "Narrow", Bmass, mean, sigma1);
    RooGaussian gauss2("gauss2", "Wide", Bmass, mean, sigma2);

    // Soma normalizada
    RooAddPdf shape("shape", "Normalized Double Gaussian", RooArgList(gauss1, gauss2), RooArgList(c1));

    // Modelo estendido com yield total
    RooExtendPdf model("model", "Extended model", shape, N);

    // Fit
    RooFitResult* result = model.fitTo(data, Range("fitRange"), Save(), Extended(kTRUE));

    // Valores Legenda
    double mean_val = mean.getVal();
    double mean_err = mean.getError();
    double N_val = N.getVal();
    double N_err = N.getError();

    //std::cout << "CANDIDATES: " << h->GetEntries() << std:endl;

    // Desenhar
    TCanvas* c = new TCanvas("c_roofit", "RooFit", 800, 600);
    RooPlot* frame = Bmass.frame();
    frame->SetTitle(" ");

    // Dar Nome ao Eixo dos xx Dependendo da Partícula
    frame->GetXaxis()->SetTitle(X_Axis_Title);
    frame->GetYaxis()->SetTitle(Form("Events / ( %.4f )", interval_of_bins));
    frame->GetYaxis()->SetTitleOffset(1.35);

    data.plotOn(frame, MarkerStyle(20), MarkerSize(1.0), Name("h_roofit")); // Circle marker
    model.plotOn(frame, LineColor(kBlue), Name("model"), Range("fitRange"), NormRange("fitRange")); // Blue model line
    shape.plotOn(frame, Components(gauss1), LineStyle(kDashed), LineColor(kGreen + 2), Range("fitRange"), NormRange("fitRange"));
    shape.plotOn(frame, Components(gauss2), LineStyle(kDashed), LineColor(kRed), Range("fitRange"), NormRange("fitRange"));
    
    frame->Draw();

    int nFitParams = result->floatParsFinal().getSize();
    double chi2 = frame->chiSquare("model", "h_roofit", nFitParams);
    
    // Legend 1 — MC Signal, Model, Number of Candidates
    auto legend_signal = new TLegend(x_min_legend, 0.76, x_max_legend, 0.88);
    legend_signal->SetTextSize(0.03);
    legend_signal->SetTextFont(42);
    legend_signal->AddEntry((TObject*)frame->findObject("h_roofit"), legend_name + " - MC Signal", "p");
    legend_signal->AddEntry((TObject*)frame->findObject("model"), "Model: Double Gaussian", "l");
    legend_signal->AddEntry((TObject*)0, Form("No. of MC Candidates = %.0f", h->GetEntries()), "");
    legend_signal->Draw();

    // TPaveText for fit parameters (more compact, no indent)
    TPaveText* pave = new TPaveText(x_min_legend, 0.47, x_max_legend, 0.75, "NDC");
    pave->SetTextAlign(12); // left align
    pave->SetTextFont(42);
    pave->SetTextSize(0.03);
    pave->SetFillColor(0);
    pave->SetBorderSize(1);

    pave->AddText(Form("Mean = %.5f #pm %.5f", mean.getVal(), mean.getError()));
    pave->AddText(Form("#sigma_{1} = %.5f #pm %.5f", sigma1.getVal(), sigma1.getError()));
    pave->AddText(Form("#sigma_{2} = %.5f #pm %.5f", sigma2.getVal(), sigma2.getError()));
    pave->AddText(Form("c_{1} = %.4f #pm %.4f", c1.getVal(), c1.getError()));
    pave->AddText(Form("N = %.2f #pm %.2f", N.getVal(), N.getError()));
    pave->AddText(Form("#chi^{2}/ndf = %.2f", chi2));
    pave->Draw();

    // Terminal
    std::cout << " ➤ Mean = " << mean.getVal() << std::endl;
    std::cout << " ➤ Sigma1 = " << sigma1.getVal() << std::endl;
    std::cout << " ➤ Sigma2 = " << sigma2.getVal() << std::endl;
    std::cout << " ➤ c1 (G1 fraction) = " << c1.getVal() << std::endl;
    std::cout << " ➤ N = " << N_val << std::endl;
    std::cout << " ➤ Total entries = " << h->GetEntries() << std::endl;

    // Guardar com Nome Específico da Partícula
    c->SaveAs(file_name);

    // Limpar
    delete h;
    delete c;
}










void fit_mc_signal_Bs() {
    TTree*  tree          = nullptr;
    TString path_to_file  = "/lstore/cms/hlegoinha/DATA_Sharing/Bmesons/MC_Bs.root";
    TString file_name     = "MC_Bs_DoubleGaussian_Fit.pdf";
    TString path_to_tree  = "ntphi";
    TString X_Axis_Title  = "m_{J/#Psi #Phi} [GeV/c^{2}]";
    TString legend_name   = "B_{s}";

    // Histogram binning
    double xlow = 5.0;
    double xhigh = 6.0;
    double interval_of_bins = 0.0075;
    int nbins = int((xhigh - xlow) / interval_of_bins);

    // Legend position
    double x_min_legend = 0.50;
    double x_max_legend = 0.88;

    // Open file and tree
    TFile* file = TFile::Open(path_to_file);
    if (!file || file->IsZombie()) {
        std::cerr << "Ficheiro de MC não encontrado!" << std::endl;
        return;
    }
    file->GetObject(path_to_tree, tree);
    if (!tree) {
        std::cerr << "Tree não encontrada!" << std::endl;
        file->Close();
        delete file;
        return;
    }

    TString cut = Form("(%s) && (%s) && (%s) && (%s)",
        isMCsignal.Data(),
        ACCcuts_ppRef.Data(),
        SELcuts_ppRef.Data(),
        TRGmatching.Data()
    );

    // Fill histogram
    TH1F* h = new TH1F("h_roofit", "MC Signal;Bmass [GeV/c^{2}];Entries", nbins, xlow, xhigh);
    tree->Draw("Bmass >> h_roofit", cut, "goff");

    if (h->GetEntries() == 0) {
        std::cerr << "Histograma vazio após os cortes!" << std::endl;
        delete h;
        file->Close();
        delete file;
        return;
    }

    double interval_fit_min = 5.26;
    double interval_fit_max = 5.46;

    double mean_init = 5.36;
    double mean_min  = 5.32;
    double mean_max  = 5.40;

    double sigma1_init = 0.036; 
    double sigma1_min = 0.001; 
    double sigma1_max = 0.09;

    double sigma2_init = 0.011; 
    double sigma2_min = 0.001; 
    double sigma2_max = 0.07;

    double sigma3_init = 0.020; 
    double sigma3_min = 0.001; 
    double sigma3_max = 0.05;

    double c1_init = 0.52; 
    double c1_min = 0.00; 
    double c1_max = 1.00;
    
    double c2_init = 0.25; 
    double c2_min = 0.00; 
    double c2_max = 1.00; 

    double N_min = 0;
    double N_max = 900000;

    // === RooFit objects ===
    RooRealVar Bmass("Bmass", "Bmass", xlow, xhigh);
    Bmass.setRange("fitRange", interval_fit_min, interval_fit_max);

    RooDataHist data("data", "Histograma RooFit", RooArgList(Bmass), h);

    RooRealVar mean  ("mean",  "mean",  mean_init,  mean_min,  mean_max);
    RooRealVar sigma1("sigma1","sigma1",sigma1_init,sigma1_min,sigma1_max);
    RooRealVar sigma2("sigma2","sigma2",sigma2_init,sigma2_min,sigma2_max);
    //RooRealVar sigma3("sigma3","sigma3",sigma3_init,sigma3_min,sigma3_max);

    // Fractions (the third fraction is 1 - c1 - c2)
    RooRealVar c1("c1", "Fraction of G1", c1_init, c1_min, c1_max);
    //RooRealVar c2("c2", "Fraction of G2", c2_init, c2_min, c2_max);

    RooRealVar N("N", "Total yield", h->GetEntries(), N_min, N_max);

    // Three Gaussians with shared mean
    RooGaussian gauss1("gauss1", "Component 1", Bmass, mean, sigma1);
    RooGaussian gauss2("gauss2", "Component 2",    Bmass, mean, sigma2);
    //RooGaussian gauss3("gauss3", "Component 3",   Bmass, mean, sigma3);

    // Triple Gaussian as normalized shape:
    // RooAddPdf with 3 PDFs and 2 fractions => third fraction is implicit (1 - c1 - c2)
    RooAddPdf shape("shape", "Normalized Double Gaussian",
                    RooArgList(gauss1, gauss2),
                    RooArgList(c1));

    // Extended model with total yield
    RooExtendPdf model("model", "Extended model", shape, N);

    // Fit in the defined range
    RooFitResult* result = model.fitTo(data, Range("fitRange"), Save(), Extended(kTRUE));

    // === Plot ===
    TCanvas* c = new TCanvas("c_roofit", "RooFit", 800, 600);
    RooPlot* frame = Bmass.frame();
    frame->SetTitle(" ");
    frame->GetXaxis()->SetTitle(X_Axis_Title);
    frame->GetYaxis()->SetTitle(Form("Events / ( %.4f )", interval_of_bins));
    frame->GetYaxis()->SetTitleOffset(1.35);

    data.plotOn  (frame, MarkerStyle(20), MarkerSize(1.0), Name("h_roofit"));
    model.plotOn (frame, LineColor(kBlue), Name("model"), Range("fitRange"), NormRange("fitRange"));
    shape.plotOn (frame, Components(gauss1), LineStyle(kDashed), LineColor(kGreen+2), Range("fitRange"), NormRange("fitRange"));
    shape.plotOn (frame, Components(gauss2), LineStyle(kDashed), LineColor(kRed),     Range("fitRange"), NormRange("fitRange"));
    //shape.plotOn (frame, Components(gauss3), LineStyle(kDashed), LineColor(kMagenta+1), Range("fitRange"), NormRange("fitRange"));

    frame->Draw();

    int nFitParams = result->floatParsFinal().getSize();
    double chi2 = frame->chiSquare("model", "h_roofit", nFitParams);

    // Legend
    auto legend_signal = new TLegend(x_min_legend, 0.76, x_max_legend, 0.88);
    legend_signal->SetTextSize(0.03);
    legend_signal->SetTextFont(42);
    legend_signal->AddEntry((TObject*)frame->findObject("h_roofit"), legend_name + " - MC Signal", "p");
    legend_signal->AddEntry((TObject*)frame->findObject("model"), "Model: Double Gaussian", "l");
    legend_signal->AddEntry((TObject*)0, Form("No. of MC Candidates = %.0f", h->GetEntries()), "");
    legend_signal->Draw();

    // Fit parameter box
    TPaveText* pave = new TPaveText(x_min_legend, 0.44, x_max_legend, 0.75, "NDC");
    pave->SetTextAlign(12);
    pave->SetTextFont(42);
    pave->SetTextSize(0.03);
    pave->SetFillColor(0);
    pave->SetBorderSize(1);
    pave->AddText(Form("Mean = %.5f #pm %.5f", mean.getVal(), mean.getError()));
    pave->AddText(Form("#sigma_{1} = %.5f #pm %.5f", sigma1.getVal(), sigma1.getError()));
    pave->AddText(Form("#sigma_{2} = %.5f #pm %.5f", sigma2.getVal(), sigma2.getError()));
    //pave->AddText(Form("#sigma_{3} = %.5f #pm %.5f", sigma3.getVal(), sigma3.getError()));
    pave->AddText(Form("c_{1} = %.4f #pm %.4f", c1.getVal(), c1.getError()));
    //pave->AddText(Form("c_{2} = %.4f #pm %.4f", c2.getVal(), c2.getError()));
    pave->AddText(Form("N = %.2f #pm %.2f", N.getVal(), N.getError()));
    pave->AddText(Form("#chi^{2}/ndf = %.2f", chi2));
    pave->Draw();

    // Terminal output
    std::cout << " ➤ Mean = " << mean.getVal() << std::endl;
    std::cout << " ➤ Sigma1 = " << sigma1.getVal() << std::endl;
    std::cout << " ➤ Sigma2 = " << sigma2.getVal() << std::endl;
    //std::cout << " ➤ Sigma3 = " << sigma3.getVal() << std::endl;
    std::cout << " ➤ c1 (G1 fraction) = " << c1.getVal() << std::endl;
    //std::cout << " ➤ c2 (G2 fraction) = " << c2.getVal() << std::endl;
    std::cout << " ➤ N = " << N.getVal() << std::endl;
    std::cout << " ➤ Total entries = " << h->GetEntries() << std::endl;

    // Save plot
    c->SaveAs(file_name);

    // Cleanup
    delete h;
    delete c;
    file->Close();
    delete file;
}







// B0 com RT e WT
void fit_bd_rt_wt() {
    using namespace RooFit;

    // ----------------------------
    // I/O and plotting config (Bd)
    // ----------------------------
    TString path_to_file = "/lstore/cms/lekai/Bmeson/MC/ppRef/Bd_phat5_Bfinder.root";
    TString path_to_tree = "Bfinder/ntKstar";
    TString X_Axis_Title = "m_{J/#Psi K^{*}} [GeV/c^{2}]";
    TString legend_name  = "B^{0}";

    // Histogram binning and fit range (match your Bd setup)
    const double xlow = 5.0;
    const double xhigh = 6.0;
    const double interval_of_bins = 0.0075;
    int nbins = int((xhigh - xlow) / interval_of_bins);

    // Fit subrange (as in your Bd config)
    const double fitMin = 5.0;
    const double fitMax = 6.0;

    // Legend positions (as in your Bd config)
    const double x_min_legend = 0.50;
    const double x_max_legend = 0.95;

    // ----------------------------
    // Open file & get tree
    // ----------------------------
    TFile* file = TFile::Open(path_to_file);
    if (!file || file->IsZombie()) {
        std::cerr << "Ficheiro de MC não encontrado!" << std::endl;
        return;
    }
    TTree* tree = nullptr;
    file->GetObject(path_to_tree, tree);
    if (!tree) {
        std::cerr << "Tree não encontrada!" << std::endl;
        file->Close(); delete file;
        return;
    }

    // --------------------------------------
    // RooFit observable & common definitions
    // --------------------------------------
    RooRealVar Bmass("Bmass", "Bmass", xlow, xhigh);
    Bmass.setRange("fitRange", fitMin, fitMax);

    // Helper lambda to make a RooDataHist from a tree+cut
    auto makeData = [&](const char* hname, const TString& cut)->std::pair<TH1F*, RooDataHist*> {
        TH1F* h = new TH1F(hname, "MC;Bmass [GeV/c^{2}];Entries", nbins, xlow, xhigh);
        h->Sumw2();
        tree->Draw(Form("Bmass >> %s", hname), cut, "goff");
        if (h->GetEntries() == 0) {
            std::cerr << "Histograma vazio após os cortes para " << hname << "!" << std::endl;
            delete h;
            return {nullptr, nullptr};
        }
        RooDataHist* data = new RooDataHist((TString(hname) + "_data").Data(), "RooDataHist", RooArgList(Bmass), h);
        return {h, data};
    };

    // Helper lambda to decorate a frame, draw pulls, & save a canvas
    auto savePlot = [&](RooPlot* frame,
                        const char* pdfName,
                        double binw,
                        double chi2,
                        const std::vector<std::pair<TString, TString>>& entries,
                        const std::vector<TString>& extraLines,
                        const char* dataObjName,
                        const char* modelObjName) {
        TCanvas* c = new TCanvas((TString("c_") + pdfName).Data(), "RooFit", 800, 800);

        // Two pads: top = fit, bottom = pulls
    TPad* p1 = new TPad("p1","p1", 0.0, 0.15, 1.0, 1.0); 
    p1->SetBottomMargin(0.04);
    p1->SetLeftMargin(0.12);
    p1->SetRightMargin(0.04);
    p1->Draw();
    p1->cd();


        // Axes/labels for top frame (hide x labels on top pad)
        frame->GetXaxis()->SetTitle(X_Axis_Title);
        frame->GetYaxis()->SetTitle(Form("Events / ( %.4f )", binw));
        frame->GetYaxis()->SetTitleOffset(1.35);
        frame->GetXaxis()->SetLabelSize(0);

        frame->Draw();

        // Default TPaveText height
        double pave_ymin = 0.41;
        double pave_ymax = 0.72;

        // Make the params box taller for the combined RT+WT figure
        if (TString(pdfName) == "Bd_RT_WT_combined_fit.pdf") {
            pave_ymin = 0.24;
            pave_ymax = 0.72;
        }

        // Legend block (top pad)
        auto leg = new TLegend(x_min_legend, 0.72, x_max_legend, 0.88);
        leg->SetTextSize(0.03);
        leg->SetTextFont(42);
        for (auto& kv : entries) {
            TObject* obj = nullptr;
            const char* drawOpt = "l";  // default: line sample
            if (kv.first.Length()) {
                obj = (TObject*)frame->findObject(kv.first);
                if (kv.first.Contains("data")) drawOpt = "pe";  // data_* -> marker + errors
                leg->AddEntry(obj, kv.second, drawOpt);
            } else {
                leg->AddEntry((TObject*)0, kv.second, "");
            }
        }
        leg->Draw();

        // Fit parameter box (top pad)
        TPaveText* pave = new TPaveText(x_min_legend, pave_ymin, x_max_legend, pave_ymax, "NDC");
        pave->SetTextAlign(12);
        pave->SetTextFont(42);
        pave->SetTextSize(0.03);
        pave->SetFillColor(0);
        pave->SetBorderSize(1);
        for (auto& line : extraLines) {
            pave->AddText(line);
        }
        pave->AddText(Form("#chi^{2}/ndf = %.2f", chi2));
        pave->Draw();

        // ---- Bottom pad: pulls ----
        c->cd();
        TPad* p2 = new TPad("p2","p2", 0.0, 0.0, 1.0, 0.15);
        p2->SetTopMargin(0.02);
        p2->SetBottomMargin(0.38);
        p2->SetLeftMargin(0.12);
        p2->SetRightMargin(0.04);
        p2->Draw();
        p2->cd();

        RooPlot* pullFrame = Bmass.frame();
        RooHist* pullHist = frame->pullHist(dataObjName, modelObjName); // names must match
        pullHist->SetMarkerSize(0.7);
        pullHist->SetLineWidth(0);
        for (int i = 0; i < pullHist->GetN(); ++i) {
            pullHist->SetPointEXlow(i, 0);
            pullHist->SetPointEXhigh(i, 0);
            pullHist->SetPointEYlow(i, 0);
            pullHist->SetPointEYhigh(i, 0);
        }
        pullFrame->addPlotable(pullHist, "P");

        pullFrame->SetTitle("");
        pullFrame->GetYaxis()->SetTitle("Pulls");
        pullFrame->GetYaxis()->SetNdivisions(505);
        pullFrame->GetYaxis()->SetTitleSize(0.13); 
        pullFrame->GetYaxis()->SetTitleOffset(0.30); 
        pullFrame->GetYaxis()->SetLabelSize(0.12);

        pullFrame->GetXaxis()->SetTitle(X_Axis_Title);
        pullFrame->GetXaxis()->SetTitleSize(0.14);    
        pullFrame->GetXaxis()->SetTitleOffset(1.00);
        pullFrame->GetXaxis()->SetLabelSize(0.12);

        pullFrame->SetMinimum(-3.5);
        pullFrame->SetMaximum(3.5);
        pullFrame->Draw();

        TLine* zeroLine = new TLine(Bmass.getMin(), 0, Bmass.getMax(), 0);
        zeroLine->SetLineColor(kBlue);
        zeroLine->SetLineStyle(1);
        zeroLine->SetLineWidth(1);
        zeroLine->Draw("same");

        c->SaveAs(pdfName);
        delete c;
    };


    // ===========================================================================================
    // FIT 1: isMCsignal && ACCcuts_ppRef && SELcuts_ppRef && TRGmatching  (Double Gaussian)
    // ===========================================================================================
    TString cut_RT = Form("(%s) && (%s) && (%s) && (%s)",
                          isMCsignal.Data(),
                          ACCcuts_ppRef.Data(),
                          SELcuts_ppRef.Data(),
                          TRGmatching.Data());

    auto [h_RT, data_RT] = makeData("h_RT", cut_RT);
    if (!h_RT) { file->Close(); delete file; return; }

    // Initial parameters (take from your Bd setup)
    RooRealVar mean("mean", "Mean", 5.28, 5.26, 5.30);
    RooRealVar sigma1_1("sigma1_1", "Sigma1_1", 0.014, 0.0001, 0.2);
    RooRealVar sigma2_1("sigma2_1", "Sigma2_1", 0.034, 0.0001, 0.2);
    RooRealVar c1_1("c1_1", "c1_1 (RT fraction of G1)", 0.80, 0.01, 0.99);
    RooRealVar N_1("N_1", "Yield RT", h_RT->GetEntries(), 0, 900000);

    RooGaussian g1_RT("g1_RT", "RT G1", Bmass, mean, sigma1_1);
    RooGaussian g2_RT("g2_RT", "RT G2", Bmass, mean, sigma2_1);
    RooAddPdf   shape_RT("shape_RT", "RT DoubleG", RooArgList(g1_RT, g2_RT), RooArgList(c1_1));
    RooExtendPdf model_RT("model_RT", "RT Extended", shape_RT, N_1);


    // ===========================================================================================
    // FIT 2: (Bgen==41000) && ACCcuts_ppRef && SELcuts_ppRef && TRGmatching (Double Gaussian)
    //        Simultaneous with RT: Mean is shared and floats in the sim fit
    // ===========================================================================================
    TString cut_WT = Form("(Bgen==41000) && (%s) && (%s) && (%s)",
                          ACCcuts_ppRef.Data(),
                          SELcuts_ppRef.Data(),
                          TRGmatching.Data());

    auto [h_WT, data_WT] = makeData("h_WT", cut_WT);
    if (!h_WT) {
        delete h_RT; delete data_RT;
        file->Close(); delete file;
        return;
    }

    // WT parameters: Gaussian + Asymmetric Gaussian (bifurcated), share the same fixed Mean
    RooRealVar sigmaG_2("sigmaG_2", "SigmaG_2 (WT Gauss)", 0.014, 0.0001, 0.5);
    RooRealVar sigmaL_2("sigmaL_2", "SigmaL_2 (WT AsymG left)", 0.020, 0.0001, 0.5);
    RooRealVar sigmaR_2("sigmaR_2", "SigmaR_2 (WT AsymG right)", 0.040, 0.0001, 0.5);
    RooRealVar c1_2("c1_2", "c1_2 (WT fraction of Gauss)", 0.50, 0.01, 0.99);
    RooRealVar N_2("N_2", "Yield WT", h_WT->GetEntries(), 0, 900000);

    RooGaussian    gG_WT("gG_WT", "WT Gaussian", Bmass, mean, sigmaG_2); // mean fixed from Fit 1
    RooBifurGauss  bif_WT("bif_WT", "WT AsymGaussian", Bmass, mean, sigmaL_2, sigmaR_2);
    RooAddPdf      shape_WT("shape_WT", "WT Gauss+AsymG", RooArgList(gG_WT, bif_WT), RooArgList(c1_2));
    RooExtendPdf   model_WT("model_WT", "WT Extended", shape_WT, N_2);

    // === SIMULTANEOUS FIT (RT + WT share the same Mean) ===
    RooCategory sample("sample","sample");
    sample.defineType("RT");
    sample.defineType("WT");

    // Build combined (binned) dataset with category index
    RooDataHist combData("combData","combined RT+WT", RooArgList(Bmass),
                        Index(sample),
                        Import("RT", *data_RT),
                        Import("WT", *data_WT));

    // Simultaneous PDF: one per category
    RooSimultaneous simPdf("simPdf","simultaneous RT+WT", sample);
    simPdf.addPdf(model_RT, "RT");
    simPdf.addPdf(model_WT, "WT");

    // Fit both channels at once; Mean is shared because it's the same RooRealVar in both PDFs
    RooFitResult* simRes = simPdf.fitTo(combData, Range("fitRange"), Save(), Extended(kTRUE), PrintLevel(-1));

    // Freeze shapes after the simultaneous fit (for later use and for the final combined fit)
    mean.setConstant(true);
    sigma1_1.setConstant(true);
    sigma2_1.setConstant(true);
    c1_1.setConstant(true);

    sigmaG_2.setConstant(true);
    sigmaL_2.setConstant(true);
    sigmaR_2.setConstant(true);
    c1_2.setConstant(true);



    // Plot FIT 1
    RooPlot* fr1 = Bmass.frame(Title("RT (isMCsignal) fit"));
    data_RT->plotOn(fr1, MarkerStyle(20), MarkerSize(1.0), Name("data_RT"));
    model_RT.plotOn(fr1, LineColor(kBlue), Name("model_RT"), Range("fitRange"), NormRange("fitRange"));
    shape_RT.plotOn(fr1, Components(g1_RT), LineStyle(kDashed), LineColor(kGreen+2), Range("fitRange"), NormRange("fitRange"));
    shape_RT.plotOn(fr1, Components(g2_RT), LineStyle(kDashed), LineColor(kRed),     Range("fitRange"), NormRange("fitRange"));

    const double binw_RT = h_RT->GetBinWidth(1);
    int nPars_RT = simRes ? simRes->floatParsFinal().getSize() : 0;
    double chi2_RT = fr1->chiSquare("model_RT", "data_RT", nPars_RT);
    {
        std::vector<std::pair<TString,TString>> legEntries = {
            {"data_RT", legend_name + " - MC Signal"},
            {"model_RT", "Model: Double Gaussian"},
            {"", Form("No. of MC Candidates = %.0f", h_RT->GetEntries())}
        };
        std::vector<TString> lines = {
            Form("Mean (shared) = %.5f #pm %.5f", mean.getVal(), mean.getError()),
            Form("#sigma_{1,RT} = %.5f #pm %.5f", sigma1_1.getVal(), sigma1_1.getError()),
            Form("#sigma_{2,RT} = %.5f #pm %.5f", sigma2_1.getVal(), sigma2_1.getError()),
            Form("c_{1,RT} = %.4f #pm %.4f", c1_1.getVal(), c1_1.getError()),
            Form("N_{1} = %.0f #pm %.0f", N_1.getVal(), N_1.getError())
        };
        savePlot(fr1, "Bd_RT_fit.pdf", binw_RT, chi2_RT, legEntries, lines, "data_RT", "model_RT");
    }


    // Plot FIT 2
    RooPlot* fr2 = Bmass.frame(Title("WT (Bgen==41000) fit"));
    data_WT->plotOn(fr2, MarkerStyle(20), MarkerSize(1.0), Name("data_WT"));
    model_WT.plotOn(fr2, LineColor(kBlue), Name("model_WT"), Range("fitRange"), NormRange("fitRange"));
    shape_WT.plotOn(fr2, Components(gG_WT),   LineStyle(kDashed), LineColor(kGreen+2), Range("fitRange"), NormRange("fitRange"));
    shape_WT.plotOn(fr2, Components(bif_WT),  LineStyle(kDashed), LineColor(kRed),     Range("fitRange"), NormRange("fitRange"));



    const double binw_WT = h_WT->GetBinWidth(1);
    int nPars_WT = simRes ? simRes->floatParsFinal().getSize() : 0;
    double chi2_WT = fr2->chiSquare("model_WT", "data_WT", nPars_WT);
    {
        std::vector<std::pair<TString,TString>> legEntries = {
            {"data_WT", legend_name + " - MC WT"},
            {"model_WT", "Model: Gauss + Asym Gauss"},
            {"", Form("No. of MC Candidates = %.0f", h_WT->GetEntries())}
        };
    std::vector<TString> lines = {
        Form("Mean (shared) = %.5f #pm %.5f", mean.getVal(), mean.getError()),
        Form("#sigma_{G,WT} = %.5f #pm %.5f", sigmaG_2.getVal(), sigmaG_2.getError()),
        Form("#sigma_{L,WT} = %.5f #pm %.5f", sigmaL_2.getVal(), sigmaL_2.getError()),
        Form("#sigma_{R,WT} = %.5f #pm %.5f", sigmaR_2.getVal(), sigmaR_2.getError()),
        Form("c_{1 WT} = %.4f #pm %.4f", c1_2.getVal(), c1_2.getError()),
        Form("N_{WT} = %.0f #pm %.0f", N_2.getVal(), N_2.getError())
    };

        savePlot(fr2, "Bd_WT_fit.pdf", binw_WT, chi2_WT, legEntries, lines, "data_WT", "model_WT");
    }

    // ===========================================================================================
    // FIT 3: Combined ((isMCsignal)||(Bgen==41000)) && ACC && SEL && TRGmatching
    //        Four Gaussians, Mean fixed, (σ1_1,σ2_1,c1_1) fixed for RT, (σ1_2,σ2_2,c1_2) fixed for WT
    //        Separate yields N_RT and N_WT (requested)
    // ===========================================================================================
    TString cut_ALL = Form("((%s)||(Bgen==41000)) && (%s) && (%s) && (%s)",
                           isMCsignal.Data(),
                           ACCcuts_ppRef.Data(),
                           SELcuts_ppRef.Data(),
                           TRGmatching.Data());

    auto [h_ALL, data_ALL] = makeData("h_ALL", cut_ALL);
    if (!h_ALL) {
        delete h_RT; delete data_RT;
        delete h_WT; delete data_WT;
        file->Close(); delete file;
        return;
    }

    // RT fixed-shape sub-PDF
    RooGaussian g1_RT_fix("g1_RT_fix", "RT G1 (fixed)", Bmass, mean, sigma1_1);
    RooGaussian g2_RT_fix("g2_RT_fix", "RT G2 (fixed)", Bmass, mean, sigma2_1);
    RooAddPdf   sigRT("sigRT", "RT DoubleG (fixed)", RooArgList(g1_RT_fix, g2_RT_fix), RooArgList(c1_1));

    // WT fixed-shape sub-PDF (Gauss + Asymmetric Gauss)
    RooGaussian   gG_WT_fix("gG_WT_fix", "WT Gaussian (fixed)", Bmass, mean, sigmaG_2);
    RooBifurGauss bif_WT_fix("bif_WT_fix", "WT AsymGaussian (fixed)", Bmass, mean, sigmaL_2, sigmaR_2);
    RooAddPdf     sigWT("sigWT", "WT Gauss+AsymG (fixed)", RooArgList(gG_WT_fix, bif_WT_fix), RooArgList(c1_2));


    // Separate extended yields
    RooRealVar N_RT("N_RT", "Yield RT (final)", h_RT->GetEntries(), 0, 1e7);
    RooRealVar N_WT("N_WT", "Yield WT (final)", h_WT->GetEntries(), 0, 1e7);

    // Wrap each component in RooExtend (robust extended fit with named yields)
    RooExtendPdf extRT("extRT", "Extended RT", sigRT, N_RT);
    RooExtendPdf extWT("extWT", "Extended WT", sigWT, N_WT);

    // Sum of extended components
    RooAddPdf total("total", "RT+WT total", RooArgList(extRT, extWT));

    RooFitResult* res3 = total.fitTo(*data_ALL, Range("fitRange"), Save(), Extended(kTRUE), PrintLevel(-1));

    // Plot FIT 3
    RooPlot* fr3 = Bmass.frame(Title("Combined RT+WT (fixed shapes)"));
    data_ALL->plotOn(fr3, MarkerStyle(20), MarkerSize(1.0), Name("data_ALL"));
    total.plotOn(fr3, LineColor(kBlue), Name("total"), Range("fitRange"), NormRange("fitRange"));
    total.plotOn(fr3, Components("sigRT"), LineStyle(kDashed), LineColor(kGreen+2), Range("fitRange"), NormRange("fitRange"));
    total.plotOn(fr3, Components("sigWT"), LineStyle(kDashed), LineColor(kRed),     Range("fitRange"), NormRange("fitRange"));

    const double binw_ALL = h_ALL->GetBinWidth(1);
    int nPars_ALL = res3 ? res3->floatParsFinal().getSize() : 0;
    double chi2_ALL = fr3->chiSquare("total", "data_ALL", nPars_ALL);
    {
        std::vector<std::pair<TString,TString>> legEntries = {
            {"data_ALL", legend_name + " - MC (RT+WT)"},
            {"total", "Model: 4 Gaussians"},
            {"", Form("No. of MC Candidates = %.0f", h_ALL->GetEntries())}
        };
    std::vector<TString> lines = {
        Form("Mean (shared) = %.5f #pm %.5f", mean.getVal(), mean.getError()),
        Form("#sigma_{1,RT} = %.5f #pm %.5f",  sigma1_1.getVal(), sigma1_1.getError()),
        Form("#sigma_{2,RT} = %.5f #pm %.5f",  sigma2_1.getVal(), sigma2_1.getError()),
        Form("c_{1,RT} = %.4f #pm %.4f",       c1_1.getVal(),   c1_1.getError()),
        Form("N_{RT} = %.2f #pm %.2f",         N_RT.getVal(),   N_RT.getError()),
        Form("#sigma_{1,WT} = %.5f #pm %.5f",  sigmaG_2.getVal(), sigmaG_2.getError()),
        Form("#sigma_{2L,WT} = %.5f #pm %.5f",  sigmaL_2.getVal(), sigmaL_2.getError()),
        Form("#sigma_{2R,WT} = %.5f #pm %.5f",  sigmaR_2.getVal(), sigmaR_2.getError()),
        Form("c_{1,WT} = %.4f #pm %.4f",       c1_2.getVal(),   c1_2.getError()),
        Form("N_{WT} = %.2f #pm %.2f",         N_WT.getVal(),   N_WT.getError())
    };
        savePlot(fr3, "Bd_RT_WT_combined_fit.pdf", binw_ALL, chi2_ALL, legEntries, lines, "data_ALL", "total");
    }

    // ----------------------------
    // Cleanup
    // ----------------------------
    delete h_RT;  delete data_RT;
    delete h_WT;  delete data_WT;
    delete h_ALL; delete data_ALL;

    file->Close(); delete file;

    // Terminal printout
    std::cout << "=== Fit 1 (RT/isMCsignal) ===" << std::endl;
    std::cout << " Mean = " << mean.getVal() << " +- " << mean.getError() << std::endl;
    std::cout << " Sigma1_1 = " << sigma1_1.getVal() << " +- " << sigma1_1.getError() << std::endl;
    std::cout << " Sigma2_1 = " << sigma2_1.getVal() << " +- " << sigma2_1.getError() << std::endl;
    std::cout << " c1_1 = " << c1_1.getVal() << " +- " << c1_1.getError() << std::endl;
    std::cout << " N_1 = " << N_1.getVal() << " +- " << N_1.getError() << std::endl;

    std::cout << "=== Fit 2 (WT/Bgen==41000) ===" << std::endl;
    std::cout << " Mean = " << mean.getVal() << std::endl;

    std::cout << " SigmaG_2 = " << sigmaG_2.getVal() << " +- " << sigmaG_2.getError() << std::endl;
    std::cout << " SigmaL_2 = " << sigmaL_2.getVal() << " +- " << sigmaL_2.getError() << std::endl;
    std::cout << " SigmaR_2 = " << sigmaR_2.getVal() << " +- " << sigmaR_2.getError() << std::endl;
    std::cout << " c1_2 (Gauss frac) = " << c1_2.getVal() << " +- " << c1_2.getError() << std::endl;

    std::cout << " N_2 = " << N_2.getVal() << " +- " << N_2.getError() << std::endl;

    std::cout << "=== Fit 3 (Combined RT+WT) ===" << std::endl;
    std::cout << " N_RT = " << N_RT.getVal() << " +- " << N_RT.getError() << std::endl;
    std::cout << " N_WT = " << N_WT.getVal() << " +- " << N_WT.getError() << std::endl;
}



















void mc_double_gaussian_fit() {
    fit_mc_signal_roofit("Bu");       // B+ MC
    //fit_mc_signal_roofit("X3872");    // X(3872) MC
    //fit_mc_signal_roofit("PSI2S");    // PSI(2S) MC
    //fit_mc_signal_Bs();               // Bs MC
    //fit_bd_rt_wt();                     // B0 MC
}
