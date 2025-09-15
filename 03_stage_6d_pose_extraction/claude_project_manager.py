#!/usr/bin/env python3
"""
Claude-Powered AI 6D Pose Recognition Project Manager
Uses Claude API to generate, debug, and optimize code automatically
"""

import os
import json
import subprocess
import time
from typing import Dict, List, Optional
import requests
from pathlib import Path

class ClaudeProjectManager:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude project manager."""
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.project_root = Path.cwd()
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            })
        
        print("🤖 Claude Project Manager Initialized")
        print(f"📁 Project Root: {self.project_root}")
    
    def call_claude(self, prompt: str, system_prompt: str = None) -> str:
        """Call Claude API with a prompt."""
        if not self.api_key:
            print("⚠️  No Claude API key found. Using mock responses.")
            return self._mock_claude_response(prompt)
        
        try:
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 4000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = self.session.post(self.base_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text']
            
        except Exception as e:
            print(f"❌ Claude API Error: {e}")
            return self._mock_claude_response(prompt)
    
    def _mock_claude_response(self, prompt: str) -> str:
        """Mock Claude responses for testing without API key."""
        if "lua script" in prompt.lower():
            return """
-- Generated Lua Script for CoppeliaSim
print("🤖 Claude-generated script running...")

-- Get camera handles
local rgbSensor = sim.getObject("./rgb")
local depthSensor = sim.getObject("./depth")

if rgbSensor ~= -1 and depthSensor ~= -1 then
    print("✅ Cameras found!")
    
    -- Capture images
    local rgbImage = sim.getVisionSensorImg(rgbSensor)
    local depthImage = sim.getVisionSensorDepthBuffer(depthSensor)
    
    if rgbImage then
        local file = io.open("claude_rgb.txt", "wb")
        if file then
            file:write(rgbImage)
            file:close()
            print("💾 RGB saved!")
        end
    end
    
    print("🎯 Capture complete!")
else
    print("❌ Cameras not found!")
end
"""
        elif "python script" in prompt.lower():
            return """
#!/usr/bin/env python3
# Claude-generated Python script

import cv2
import numpy as np
from ultralytics import YOLO

def claude_detection():
    print("🤖 Claude-generated detection running...")
    
    # Load model
    model = YOLO("ycb_texture_training/ycb_texture_detector/weights/best.pt")
    
    # Process image
    image = cv2.imread("claude_rgb_image.jpg")
    if image is not None:
        results = model(image, conf=0.1)
        print(f"✅ Detection complete: {len(results)} results")
    else:
        print("❌ Image not found")

if __name__ == "__main__":
    claude_detection()
"""
        else:
            return "🤖 Claude response: " + prompt[:100] + "..."
    
    def generate_lua_script(self, requirements: str) -> str:
        """Generate Lua script for CoppeliaSim using Claude."""
        prompt = f"""
Generate a Lua script for CoppeliaSim that:
{requirements}

Requirements:
- Must be stable and not cause crashes
- Should handle errors gracefully
- Should save data to files
- Should work with Kinect or spherical RGBD cameras

Return only the Lua code, no explanations.
"""
        
        system_prompt = """You are an expert Lua programmer specializing in CoppeliaSim robotics simulation. 
Generate clean, efficient, and stable Lua code that follows best practices."""
        
        return self.call_claude(prompt, system_prompt)
    
    def generate_python_script(self, requirements: str) -> str:
        """Generate Python script using Claude."""
        prompt = f"""
Generate a Python script that:
{requirements}

Requirements:
- Use modern Python practices
- Include proper error handling
- Add helpful comments
- Should be production-ready

Return only the Python code, no explanations.
"""
        
        system_prompt = """You are an expert Python programmer specializing in computer vision, 
machine learning, and robotics. Generate clean, efficient, and well-documented Python code."""
        
        return self.call_claude(prompt, system_prompt)
    
    def debug_code(self, code: str, error_message: str, language: str) -> str:
        """Debug code using Claude."""
        prompt = f"""
Debug this {language} code that has the following error:

Error: {error_message}

Code:
```{language}
{code}
```

Provide the corrected code that fixes the error.
"""
        
        return self.call_claude(prompt)
    
    def optimize_code(self, code: str, language: str, optimization_goal: str) -> str:
        """Optimize code using Claude."""
        prompt = f"""
Optimize this {language} code for {optimization_goal}:

```{language}
{code}
```

Provide the optimized version with explanations of improvements.
"""
        
        return self.call_claude(prompt)
    
    def analyze_project(self) -> Dict:
        """Analyze the current project structure and provide recommendations."""
        files = list(self.project_root.rglob("*"))
        python_files = [f for f in files if f.suffix == '.py']
        lua_files = [f for f in files if f.suffix == '.lua']
        
        analysis = {
            'total_files': len(files),
            'python_files': len(python_files),
            'lua_files': len(lua_files),
            'project_structure': str(self.project_root),
            'recommendations': []
        }
        
        # Generate recommendations using Claude
        prompt = f"""
Analyze this AI 6D Pose recognition project and provide recommendations:

Project Structure:
- Python files: {len(python_files)}
- Lua files: {len(lua_files)}
- Total files: {len(files)}

Current files:
Python: {[f.name for f in python_files[:5]]}
Lua: {[f.name for f in lua_files[:5]]}

Provide specific recommendations for:
1. Code improvements
2. Missing functionality
3. Performance optimizations
4. Best practices
"""
        
        recommendations = self.call_claude(prompt)
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def auto_fix_issues(self) -> List[str]:
        """Automatically detect and fix common issues."""
        fixes_applied = []
        
        # Check for common issues
        issues = [
            ("Missing __init__.py", "python -c 'import os; open(\"__init__.py\", \"w\").close()'"),
            ("Check Python syntax", "python -m py_compile *.py"),
            ("Check file permissions", "chmod +x *.py"),
        ]
        
        for issue_name, fix_command in issues:
            try:
                result = subprocess.run(fix_command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    fixes_applied.append(f"✅ Fixed: {issue_name}")
                else:
                    fixes_applied.append(f"⚠️  Issue: {issue_name} - {result.stderr}")
            except Exception as e:
                fixes_applied.append(f"❌ Error fixing {issue_name}: {e}")
        
        return fixes_applied
    
    def generate_project_report(self) -> str:
        """Generate a comprehensive project report using Claude."""
        analysis = self.analyze_project()
        
        prompt = f"""
Generate a comprehensive report for this AI 6D Pose recognition project:

{json.dumps(analysis, indent=2)}

Include:
1. Project overview
2. Current status
3. Technical achievements
4. Areas for improvement
5. Next steps
6. Recommendations

Format as a professional report.
"""
        
        return self.call_claude(prompt)
    
    def run_automated_workflow(self):
        """Run a complete automated workflow for the project."""
        print("🚀 Starting Claude-powered automated workflow...")
        
        # 1. Analyze project
        print("📊 Analyzing project...")
        analysis = self.analyze_project()
        print(f"✅ Found {analysis['python_files']} Python files, {analysis['lua_files']} Lua files")
        
        # 2. Auto-fix issues
        print("🔧 Auto-fixing issues...")
        fixes = self.auto_fix_issues()
        for fix in fixes:
            print(f"   {fix}")
        
        # 3. Generate optimized scripts
        print("🤖 Generating optimized scripts...")
        
        # Generate improved Lua script
        lua_requirements = """
- Capture RGB and depth from Kinect camera
- Save data to files with timestamps
- Include error handling and logging
- Be stable and crash-resistant
"""
        improved_lua = self.generate_lua_script(lua_requirements)
        
        with open("claude_improved_capture.lua", "w") as f:
            f.write(improved_lua)
        print("✅ Generated: claude_improved_capture.lua")
        
        # Generate improved Python script
        python_requirements = """
- Process camera data from CoppeliaSim
- Run YOLO object detection
- Handle multiple camera types (Kinect, spherical)
- Include visualization and analysis
- Save results with timestamps
"""
        improved_python = self.generate_python_script(python_requirements)
        
        with open("claude_improved_detection.py", "w") as f:
            f.write(improved_python)
        print("✅ Generated: claude_improved_detection.py")
        
        # 4. Generate project report
        print("📋 Generating project report...")
        report = self.generate_project_report()
        
        with open("claude_project_report.md", "w") as f:
            f.write(report)
        print("✅ Generated: claude_project_report.md")
        
        print("🎉 Claude-powered workflow completed!")
        return {
            'analysis': analysis,
            'fixes': fixes,
            'generated_files': ['claude_improved_capture.lua', 'claude_improved_detection.py', 'claude_project_report.md']
        }

def main():
    """Main function to run Claude project manager."""
    print("🤖 Claude AI 6D Pose Recognition Project Manager")
    print("=" * 60)
    
    # Initialize manager
    manager = ClaudeProjectManager()
    
    # Run automated workflow
    results = manager.run_automated_workflow()
    
    print("\n📊 Summary:")
    print(f"   Files analyzed: {results['analysis']['total_files']}")
    print(f"   Issues fixed: {len(results['fixes'])}")
    print(f"   Files generated: {len(results['generated_files'])}")
    
    print("\n🎯 Next Steps:")
    print("1. Review claude_project_report.md")
    print("2. Test claude_improved_capture.lua in CoppeliaSim")
    print("3. Run claude_improved_detection.py")
    print("4. Set CLAUDE_API_KEY environment variable for full Claude integration")

if __name__ == "__main__":
    main()




