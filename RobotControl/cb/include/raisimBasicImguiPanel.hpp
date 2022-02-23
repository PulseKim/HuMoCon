//
// Created by Jemin Hwangbo on 2/28/19.
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#ifndef RAISIMOGREVISUALIZER_RAISIMBASICIMGUIPANEL_HPP
#define RAISIMOGREVISUALIZER_RAISIMBASICIMGUIPANEL_HPP
#include "guiState.hpp"

ImFont* fontBig;
ImFont* fontMid;
ImFont* fontSmall;
bool visualizeFlag = false;

void imguiRenderCallBack() {


  ImGui::SetNextWindowPos({0, 0});
//  ImGui::SetNextWindowSize({400, 1000}, 0);
  if (!ImGui::Begin("RaiSim Application Window")) {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  auto vis = raisim::OgreVis::get();
  auto world = vis->getWorld();
  vis->getPaused() = raisim::gui::manualStepping;
  ImGui::PushFont(fontBig);
  ImGui::Text("Visualization");
  ImGui::Separator();
  ImGui::PopFont();

  ImGui::PushFont(fontMid);
  ImGui::Checkbox("Visualize reference", &visualizeFlag);
  ImGui::PopFont();

  unsigned long mask = 0;

  ImGui::End();

}


void imguiSetupCallback() {

#define HI(v)   ImVec4(0.502f, 0.075f, 0.256f, v)
#define MED(v)  ImVec4(0.455f, 0.198f, 0.301f, v)
#define LOW(v)  ImVec4(0.232f, 0.201f, 0.271f, v)
  // backgrounds (@todo: complete with BG_MED, BG_LOW)
#define BG(v)   ImVec4(0.200f, 0.220f, 0.270f, v)
  // text
#define TEXT(v) ImVec4(0.860f, 0.930f, 0.890f, v)

  auto &style = ImGui::GetStyle();
  style.Alpha = 0.8;
  style.Colors[ImGuiCol_Text]                  = TEXT(0.78f);
  style.Colors[ImGuiCol_TextDisabled]          = TEXT(0.28f);
  style.Colors[ImGuiCol_WindowBg]              = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
  style.Colors[ImGuiCol_ChildWindowBg]         = BG( 0.58f);
  style.Colors[ImGuiCol_PopupBg]               = BG( 0.9f);
  style.Colors[ImGuiCol_Border]                = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
  style.Colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  style.Colors[ImGuiCol_FrameBg]               = BG( 1.00f);
  style.Colors[ImGuiCol_FrameBgHovered]        = MED( 0.78f);
  style.Colors[ImGuiCol_FrameBgActive]         = MED( 1.00f);
  style.Colors[ImGuiCol_TitleBg]               = LOW( 1.00f);
  style.Colors[ImGuiCol_TitleBgActive]         = HI( 1.00f);
  style.Colors[ImGuiCol_TitleBgCollapsed]      = BG( 0.75f);
  style.Colors[ImGuiCol_MenuBarBg]             = BG( 0.47f);
  style.Colors[ImGuiCol_ScrollbarBg]           = BG( 1.00f);
  style.Colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.09f, 0.15f, 0.16f, 1.00f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered]  = MED( 0.78f);
  style.Colors[ImGuiCol_ScrollbarGrabActive]   = MED( 1.00f);
  style.Colors[ImGuiCol_CheckMark]             = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
  style.Colors[ImGuiCol_SliderGrab]            = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
  style.Colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
  style.Colors[ImGuiCol_Button]                = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
  style.Colors[ImGuiCol_ButtonHovered]         = MED( 0.86f);
  style.Colors[ImGuiCol_ButtonActive]          = MED( 1.00f);
  style.Colors[ImGuiCol_Header]                = MED( 0.76f);
  style.Colors[ImGuiCol_HeaderHovered]         = MED( 0.86f);
  style.Colors[ImGuiCol_HeaderActive]          = HI( 1.00f);
  style.Colors[ImGuiCol_Column]                = ImVec4(0.14f, 0.16f, 0.19f, 1.00f);
  style.Colors[ImGuiCol_ColumnHovered]         = MED( 0.78f);
  style.Colors[ImGuiCol_ColumnActive]          = MED( 1.00f);
  style.Colors[ImGuiCol_ResizeGrip]            = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
  style.Colors[ImGuiCol_ResizeGripHovered]     = MED( 0.78f);
  style.Colors[ImGuiCol_ResizeGripActive]      = MED( 1.00f);
  style.Colors[ImGuiCol_PlotLines]             = TEXT(0.63f);
  style.Colors[ImGuiCol_PlotLinesHovered]      = MED( 1.00f);
  style.Colors[ImGuiCol_PlotHistogram]         = TEXT(0.63f);
  style.Colors[ImGuiCol_PlotHistogramHovered]  = MED( 1.00f);
  style.Colors[ImGuiCol_TextSelectedBg]        = MED( 0.43f);
  // [...]
  style.Colors[ImGuiCol_ModalWindowDarkening]  = BG( 0.73f);

  style.WindowPadding            = ImVec2(6, 4);
  style.WindowRounding           = 0.0f;
  style.FramePadding             = ImVec2(5, 2);
  style.FrameRounding            = 3.0f;
  style.ItemSpacing              = ImVec2(7, 1);
  style.ItemInnerSpacing         = ImVec2(1, 1);
  style.TouchExtraPadding        = ImVec2(0, 0);
  style.IndentSpacing            = 6.0f;
  style.ScrollbarSize            = 12.0f;
  style.ScrollbarRounding        = 16.0f;
  style.GrabMinSize              = 20.0f;
  style.GrabRounding             = 2.0f;

  style.WindowTitleAlign.x = 0.50f;

  style.Colors[ImGuiCol_Border] = ImVec4(0.539f, 0.479f, 0.255f, 0.162f);
  style.FrameBorderSize = 0.0f;
  style.WindowBorderSize = 1.0f;

  ImGuiIO &io = ImGui::GetIO();
  fontBig = io.Fonts->AddFontFromFileTTF((raisim::OgreVis::get()->getResourceDir() + "/font/DroidSans.ttf").c_str(), 25.0f);
  fontMid = io.Fonts->AddFontFromFileTTF((raisim::OgreVis::get()->getResourceDir() + "/font/DroidSans.ttf").c_str(), 22.0f);
  fontSmall = io.Fonts->AddFontFromFileTTF((raisim::OgreVis::get()->getResourceDir() + "/font/DroidSans.ttf").c_str(), 16.0f);
}

#endif //RAISIMOGREVISUALIZER_RAISIMBASICIMGUIPANEL_HPP
