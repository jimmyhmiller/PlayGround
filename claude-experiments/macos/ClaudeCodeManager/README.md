# Claude Code Manager

A modern, stylish macOS application for managing Claude Code sessions and workspaces.

## Features

✨ **Modern Design System**
- Dark theme with beautiful gradients and glassmorphism effects
- Carefully crafted typography and spacing
- Smooth animations and micro-interactions
- Custom buttons and controls with hover states

🏗️ **Workspace Management**
- Add/remove workspaces with visual folder picker
- Session status indicators with real-time updates
- Start/stop Claude Code processes with one click
- Beautiful card-based interface for workspace overview

📋 **Todo Management**
- Integrated todo lists for each workspace
- Priority indicators and status badges
- Smooth animations when adding/removing tasks
- Context menus for advanced actions

🎨 **Visual Polish**
- Floating animated background orbs
- Glass morphism effects and subtle shadows
- Color-coded status indicators
- Responsive layout with proper constraints

## Design Philosophy

This app follows modern macOS design principles inspired by apps you'd see on Dribbble:

- **Glassmorphism**: Translucent panels with backdrop blur effects
- **Depth & Shadows**: Layered interface with proper elevation
- **Micro-interactions**: Hover states, smooth transitions, and delightful animations
- **Typography Hierarchy**: Clear visual hierarchy with proper font weights and sizes
- **Color System**: Carefully chosen dark theme colors with accent highlights

## Usage

### Running the App

```bash
cd ClaudeCodeManager
swift run
```

### Adding Workspaces

1. Click the **+** button in the sidebar
2. Select a directory containing your project
3. The workspace will appear as a beautiful card with status indicator

### Managing Sessions

- **Start**: Click the "Start" button on any workspace card
- **Stop**: Click "Stop" to terminate the Claude Code session
- **Status**: Watch the color-coded indicators:
  - 🟢 Green: Active session
  - ⚪ Gray: Inactive
  - 🔴 Red: Error state

### Todo Management

1. Select a workspace to view its details
2. Click **+** in the Tasks section to add todos
3. Use the status dropdowns to track progress
4. Set priorities with the color-coded indicators

## Technical Implementation

### Architecture

- **ModernSidebarViewController**: Card-based workspace list with animations
- **ModernContentViewController**: Glassmorphism main area with hero section
- **ModernTodoSectionView**: Animated todo list with smooth interactions
- **DesignSystem**: Centralized colors, typography, and styling
- **SessionManager**: Process management and data persistence

### Key Components

- **ModernButton**: Custom button with multiple styles and hover effects
- **ModernSessionCard**: Workspace cards with status indicators and actions
- **GlassBackgroundView**: Animated background with floating orbs
- **StatusIndicatorView**: Pulsing status dots with color animations

### Performance Features

- Efficient session persistence using UserDefaults
- Smooth 60fps animations using Core Animation
- Responsive layout that adapts to window resizing
- Memory-efficient view recycling for large todo lists

## Build Configuration

- **Minimum macOS**: 13.0+
- **Architecture**: Universal (Apple Silicon + Intel)
- **Window Style**: Full-size content view with transparent titlebar
- **Dependencies**: None (pure AppKit + Foundation)

## Customization

The design system can be easily customized by modifying `DesignSystem.swift`:

```swift
// Colors
static let accent = NSColor(red: 0.3, green: 0.6, blue: 1.0, alpha: 1.0)
static let background = NSColor(red: 0.05, green: 0.05, blue: 0.07, alpha: 1.0)

// Typography
static let title1 = NSFont.systemFont(ofSize: 22, weight: .semibold)

// Spacing
static let lg: CGFloat = 16
```

This creates a cohesive, beautiful experience that feels at home on modern macOS.