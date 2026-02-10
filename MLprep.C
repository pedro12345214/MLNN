#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <TSystem.h>
#include <iostream>
#include <TMath.h>
#include <vector>

void MLprep() {
    const char* infile       = "/lstore/cms/u25pedrochan/MLNN/ROOT_files/Data_pp_Bs_selected_ml_output2.root";
    const char* mcfile     = "/lstore/cms/u25pedrochan/MLNN/ROOT_files/MC_pp_Bs_selected_ml_output2.root";

    //const char* infile       = "/lstore/cms/hlegoinha/DATA_Sharing/Bmesons/Data_Bu.root";
    //const char* mcfile     = "/lstore/cms/hlegoinha/DATA_Sharing/Bmesons/MC_Bu.root";
    const char* intree_name   = "Tdata";
    const char* outpath       = "/lstore/cms/u25pedrochan/MLNN/ROOT_files/Data_pp_Bu_cut2.root";
    const char* mcoutpath       = "/lstore/cms/u25pedrochan/MLNN/ROOT_files/MC_pp_Bu_cut2.root";
    const char* outtree_name  = "Tdata";   // if empty, keep same as input Tback->data sideband background Tsignal->MC signal
    const char* mcoutree_name = "Tdata";
    
    // ---- Sideband definition (edit as needed) ----

    //MC Bs  
    //double left_lo  = 5.00,        left_hi  = 5.297591014;
    //double right_lo = 5.437168986, right_hi = 6.00;

    //MC Bu
    double left_lo  = 5.00,        left_hi  = 5.178948768;
    double right_lo = 5.380091232, right_hi = 6.00;

    // ---- Define cut string ----
    TString sbCut;
    TString nCut;
    sbCut.Form("((Bmass >= %.6f && Bmass <= %.6f) || (Bmass >= %.6f && Bmass <= %.6f))",
             left_lo, left_hi, right_lo, right_hi);

    // ---- Build finite cut (remove NaN/Inf) ----
    // Put here the float/double branches you care about being valid.
    // If a branch is integer-like, it's not necessary to include it.
    std::vector<TString> Vars = {
       "Bmass", "BQvalue", "Bchi2cl", "Bcos_dtheta", "Bdtheta", "Bnorm_svpvDistance_2D", "Bnorm_trk1Dxy", "Bnorm_trk2Dxy",
        "Bpt", "Btktkmass", "Btktkpt", "Btrk1Pt", "Btrk2Pt", "Btrk1dR", "Btrk2dR",
        "BtrkPtimb", "Bujmass", "By", "nSelectedChargedTracks", "MLscore"
    };



    TString finiteCut = "1";
    for (const auto& v : Vars) {
        finiteCut += " && TMath::Finite(" + v + ")";
    }

    // ---- Final cut ----
    TString cut = "(" + sbCut + ") && (" + finiteCut + ")";
    TString precut = "(Bnorm_svpvDistance_2D>5.404) && (" + finiteCut + ")";
    TString MLcut = "MLscore>0.9259 &&  (" + finiteCut + ")";  //Bs cut  
    TString MCcut = "Bchi2cl>0.005 && Bpt>1 && (" + finiteCut + ")";
    TString Kstar_cut;
    TString Phi_cut;

    // ---- Create output directory if needed ----
    TString outdir = gSystem->DirName(outpath);
    if (gSystem->AccessPathName(outdir)) {
        if (gSystem->mkdir(outdir, kTRUE) != 0) {
            std::cerr << "ERROR: could not create output directory: " << outdir << std::endl;
            return;
        }
        std::cout << "Created output directory: " << outdir << std::endl;
    }

    // ---- Open input ----
    TFile *fin = TFile::Open(infile, "READ");
    if (!fin || fin->IsZombie()) {
        std::cerr << "ERROR: cannot open input file: " << infile << std::endl;
        return;
    }

    TTree *tin = (TTree*)fin->Get(intree_name);
    if (!tin) {
        std::cerr << "ERROR: cannot find tree '" << intree_name << "' in file.\n"
                  << "Tip: run: root -l -q 'list_trees.C(\"" << infile << "\")'\n";
        fin->Close();
        return;
    }

    TFile *fin_mc = TFile::Open(mcfile, "READ");
    if (!fin_mc || fin_mc->IsZombie()) {
        std::cerr << "ERROR: cannot open input file: " << mcfile << std::endl;
        fin_mc->Close();   
        return;
    }

    TTree *tin_mc = (TTree*)fin_mc->Get(intree_name);
    if (!tin_mc) {
        std::cerr << "ERROR: cannot find tree '" << intree_name << "' in MC file.\n"
                  << "Tip: run: root -l -q 'list_trees.C(\"" << mcfile << "\")'\n";
        fin_mc->Close();
        return;
    }

    // ---- Keep only desired branches ----
    tin->SetBranchStatus("*", 0);
    for (const auto& v : Vars) tin->SetBranchStatus(v, 1);
    tin_mc->SetBranchStatus("*", 0);
    for (const auto& v : Vars) tin_mc->SetBranchStatus(v, 1);



    std::cout << "Data Input file  : " << infile << "\n";
    std::cout << "Data Input tree  : " << intree_name << "\n";
    std::cout << "Applied Cut         : " << cut << "\n";
    std::cout << "Data Entries before: " << tin->GetEntries() << std::endl;

    std::cout << "MC Input file  : " << mcfile << "\n";
    std::cout << "MC Input tree  : " << intree_name << "\n";
    std::cout << "MC Entries before: " << tin_mc->GetEntries() << std::endl;    

    // ---- Copy with cut and write output ----
    TFile *fout = TFile::Open(outpath, "RECREATE");
    if (!fout || fout->IsZombie()) {
        std::cerr << "ERROR: cannot create output file: " << outpath << std::endl;
        fin->Close();
        return;
    }
    TTree *tout = tin->CopyTree(MLcut);
    if (!tout) {
        std::cerr << "ERROR: CopyTree failed (cut may select 0 events)\n";
        fout->Close();
        fin->Close();
        return;
    }
    TFile *fout_mc = TFile::Open(mcoutpath, "RECREATE");
    if (!fout_mc || fout_mc->IsZombie()) {
        std::cerr << "ERROR: cannot create output file: " << mcoutpath << std::endl;
        fin_mc->Close();
        return;
    }

    TTree *tout_mc = tin_mc->CopyTree(MLcut);
    if (!tout_mc) {
        std::cerr << "ERROR: CopyTree failed (cut may select 0 events)\n";
        fout_mc->Close();
        fin_mc->Close();
        return;
    }


    // Optionally rename output tree
    TString final_outtree = (TString(outtree_name).Length() > 0) ? outtree_name : intree_name;
    tout->SetName(final_outtree);

    TString final_mctree = (TString(mcoutree_name).Length() > 0) ? mcoutree_name : intree_name;
    tout_mc->SetName(final_mctree);

    std::cout << "Data Sidebands Entries after : " << tout->GetEntries() << std::endl;
    std::cout << "MC Entries after : " << tout_mc->GetEntries() << std::endl;

    fout->cd();
    tout->Write();
    fout_mc->cd();
    tout_mc->Write();
    fout->Close();
    fout_mc->Close();
    fin_mc->Close();
    fin->Close();

    std::cout << "Wrote output: " << outpath << " with tree '" << final_outtree << "'" << std::endl;
    std::cout << "Wrote output: " << mcoutpath << " with tree '" << final_mctree << "'" << std::endl;
}
