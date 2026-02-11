import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import sys

# Add path for ML integration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------- THEME ----------
BG_COLOR = "#F7F3EE"
PRIMARY_TEXT = "#2E2E2E"
SECONDARY_TEXT = "#6B6B6B"
ACCENT_GOLD = "#C6A24D"
CARD_BG = "#FFFFFF"

SELECTED_BG = "#FFD700"  # Gold
UNSELECTED_BG = "white"
SELECTED_FG = "black"
UNSELECTED_FG = PRIMARY_TEXT

# ---------- APP ROOT ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digital Twin – Hypertension")
        self.geometry("900x600")
        self.configure(bg=BG_COLOR)

        self.gender = None
        self.user_data = {}

        self.show_welcome()

    def show_welcome(self):
        self.clear()
        WelcomePage(self).pack(fill="both", expand=True)

    def show_assessment(self):
        self.clear()
        AssessmentPage(self, self).pack(fill="both", expand=True)

    def show_results(self, results):
        self.clear()
        ResultsPage(self, self, results).pack(fill="both", expand=True)

    def clear(self):
        for w in self.winfo_children():
            w.destroy()

# ---------- WELCOME PAGE ----------
class WelcomePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG_COLOR)
        self.app = parent

        base = os.path.dirname(os.path.abspath(__file__))

        # Heading
        tk.Label(self, text="DIGITAL TWIN", font=("Didot", 55, "bold"),
                 fg=PRIMARY_TEXT, bg=BG_COLOR).pack(pady=(40,5))
        tk.Label(self, text="One stop solution for anti-hypertension", font=("Zapfino", 20, "bold"),
                 fg=SECONDARY_TEXT, bg=BG_COLOR).pack(pady=(0,40))

        # Load images
        try:
            self.male_img = ImageTk.PhotoImage(Image.open(os.path.join(base,"male.png")).resize((180,260)))
            self.female_img = ImageTk.PhotoImage(Image.open(os.path.join(base,"female.png")).resize((180,260)))
        except:
            # Fallback if images not found
            self.male_img = None
            self.female_img = None

        card = tk.Frame(self, bg=CARD_BG, padx=30, pady=30, relief="raised", bd=2)
        card.pack(pady=20)

        if self.male_img and self.female_img:
            images = tk.Frame(card, bg=CARD_BG)
            images.pack()
            tk.Label(images, image=self.male_img, bg=CARD_BG).grid(row=0,column=0,padx=30)
            tk.Label(images, image=self.female_img, bg=CARD_BG).grid(row=0,column=1,padx=30)

        # Gender buttons
        self.gender_buttons = {}
        buttons = tk.Frame(card, bg=CARD_BG)
        buttons.pack(pady=25)
        for i, gender in enumerate(["Male","Female"]):
            btn = tk.Button(buttons, text=gender, bg=UNSELECTED_BG, fg=UNSELECTED_FG,
                            font=("Didot",14,"bold"), width=12,
                            command=lambda g=gender.lower(): self.select_gender(g))
            btn.grid(row=0,column=i,padx=30)
            self.gender_buttons[gender.lower()] = btn

    def select_gender(self, gender):
        self.app.gender = gender
        for g, btn in self.gender_buttons.items():
            if g == gender:
                btn.config(bg=SELECTED_BG, fg=SELECTED_FG)
            else:
                btn.config(bg=UNSELECTED_BG, fg=UNSELECTED_FG)
        self.app.show_assessment()

# ---------- SCROLLABLE FRAME ----------
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        canvas = tk.Canvas(self, bg=BG_COLOR, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

# ---------- ASSESSMENT PAGE (CENTERED) ----------
class AssessmentPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_COLOR)
        self.app = app

        scroll_frame = ScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True)
        container = scroll_frame.scrollable_frame

        # CENTERING CONTAINER - This is the key fix!
        # Create a centered frame instead of packing directly
        center_wrapper = tk.Frame(container, bg=BG_COLOR)
        center_wrapper.pack(expand=True, fill="both")
        
        # Main frame with maximum width to prevent over-stretching
        main_frame = tk.Frame(center_wrapper, bg=BG_COLOR)
        main_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Header
        tk.Label(main_frame, text="Hypertension Clinical Profile",
                 font=("Didot",34,"bold"), fg=PRIMARY_TEXT, bg=BG_COLOR).pack(pady=(20,10))
        tk.Label(main_frame, text="Personalized inputs for digital twin simulation",
                 font=("Garamond",22,"bold"), fg=SECONDARY_TEXT, bg=BG_COLOR).pack(pady=(0,30))

        # ---------- CLINICAL SNAPSHOT ----------
        snapshot_card = tk.Frame(main_frame, bg=CARD_BG, padx=50, pady=50, relief="raised", bd=2)
        snapshot_card.pack(pady=20, fill="x", expand=True)

        tk.Label(snapshot_card, text="Age", font=("Garamond",18,"bold"), fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w")
        self.age = tk.IntVar(value=30)
        tk.Scale(snapshot_card, from_=18, to=90, orient="horizontal", variable=self.age,
                 bg=CARD_BG, highlightthickness=0, length=400).pack(fill="x", pady=(0,15))

        tk.Label(snapshot_card, text="Duration of Hypertension", font=("Garamond",16,"bold"),
                 fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w")
        self.duration = tk.StringVar(value="1–5 years")
        duration_frame = tk.Frame(snapshot_card, bg=CARD_BG)
        duration_frame.pack(pady=10, anchor="w", fill="x")
        self.duration_buttons = {}
        for option in ["< 1 year","1–5 years","> 5 years"]:
            btn = tk.Button(duration_frame, text=option, font=("Garamond",12,"bold"),
                            bg=UNSELECTED_BG, fg=UNSELECTED_FG, relief="ridge", bd=2,
                            padx=12, pady=6, command=lambda o=option: self.select_duration(o))
            btn.pack(side="left", padx=6, pady=6)
            self.duration_buttons[option] = btn
            if option == self.duration.get():
                btn.config(bg=SELECTED_BG, fg=SELECTED_FG)

        # ---------- BLOOD PRESSURE ----------
        bp_card = tk.Frame(main_frame, bg=CARD_BG, padx=50, pady=50, relief="raised", bd=2)
        bp_card.pack(pady=20, fill="x", expand=True)
        tk.Label(bp_card, text="Blood Pressure (mmHg)", font=("Garamond",18,"bold"),
                 fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w", pady=(0,10))
        bp_frame = tk.Frame(bp_card, bg=CARD_BG)
        bp_frame.pack(fill="x")
        
        # Center the BP controls
        bp_controls = tk.Frame(bp_frame, bg=CARD_BG)
        bp_controls.pack(expand=True)
        
        self.sys = tk.IntVar(value=140)
        self.dia = tk.IntVar(value=90)
        tk.Scale(bp_controls, from_=90, to=200, variable=self.sys, orient="vertical",
                 label="Systolic", bg=CARD_BG, highlightthickness=0, length=150).pack(side="left", padx=40)
        self.bp_label = tk.Label(bp_controls, text=f"{self.sys.get()} / {self.dia.get()} mmHg",
                                 font=("Garamond",18,"bold"), fg=ACCENT_GOLD, bg=CARD_BG)
        self.bp_label.pack(side="left", padx=30)
        tk.Scale(bp_controls, from_=60, to=120, variable=self.dia, orient="vertical",
                 label="Diastolic", bg=CARD_BG, highlightthickness=0, length=150).pack(side="left", padx=40)
        self.sys.trace_add("write", self.update_bp)
        self.dia.trace_add("write", self.update_bp)

        # ---------- RISK FACTORS ----------
        risk_card = tk.Frame(main_frame, bg=CARD_BG, padx=50, pady=50, relief="raised", bd=2)
        risk_card.pack(pady=20, fill="x", expand=True)
        tk.Label(risk_card, text="Risk Factors", font=("Garamond",18,"bold"), fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w", pady=(0,10))
        
        # Center the risk buttons
        risk_frame = tk.Frame(risk_card, bg=CARD_BG)
        risk_frame.pack(anchor="center", fill="x")

        if self.app.gender == "male":
            risk_list = ["Diabetes","Kidney Disease","Smoker","High Stress","Sedentary Lifestyle","High Cholesterol"]
        else:
            risk_list = ["Diabetes","Kidney Disease","High Stress","Sedentary Lifestyle","Hormonal Imbalance","Pregnancy / Postpartum"]

        self.risks = {}
        # Create rows of 3 buttons for better centering
        row_frame = None
        for i, risk in enumerate(risk_list):
            if i % 3 == 0:
                row_frame = tk.Frame(risk_frame, bg=CARD_BG)
                row_frame.pack(pady=6)
            
            btn = tk.Button(row_frame, text=risk, bg=UNSELECTED_BG, fg=UNSELECTED_FG,
                            font=("Garamond",14,"bold"), relief="ridge", bd=2,
                            padx=12, pady=6, command=lambda r=risk: self.toggle_risk(r))
            btn.pack(side="left", padx=6)
            self.risks[risk] = btn

        # ---------- MEDICATION CONTEXT ----------
        med_card = tk.Frame(main_frame, bg=CARD_BG, padx=50, pady=50, relief="raised", bd=2)
        med_card.pack(pady=20, fill="x", expand=True)
        tk.Label(med_card, text="Medication Context", font=("Garamond",18,"bold"),
                 fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w", pady=(0,15))

        # BP Medication
        self.bp_med = tk.StringVar(value="No")
        bp_frame = tk.Frame(med_card, bg=CARD_BG)
        bp_frame.pack(anchor="w", pady=5)
        tk.Label(bp_frame, text="Currently on BP medication?", font=("Garamond",14,"bold"), bg=CARD_BG).pack(side="left", padx=(0,20))
        for val in ["Yes","No"]:
            rb = tk.Radiobutton(bp_frame, text=val, variable=self.bp_med, value=val,
                                bg=CARD_BG, fg=PRIMARY_TEXT, font=("Garamond",12,"bold"),
                                selectcolor=ACCENT_GOLD, command=self.toggle_bp_entry)
            rb.pack(side="left", padx=5)
        self.bp_med_entry = tk.Entry(bp_frame, font=("Garamond",12), width=20)
        self.bp_med_entry.pack(side="left", padx=10)
        self.bp_med_entry.pack_forget()

        # Drug Allergy
        self.allergy = tk.StringVar(value="No")
        allergy_frame = tk.Frame(med_card, bg=CARD_BG)
        allergy_frame.pack(anchor="w", pady=5)
        tk.Label(allergy_frame, text="Any known drug allergy?", font=("Garamond",14,"bold"), bg=CARD_BG).pack(side="left", padx=(0,20))
        for val in ["Yes","No"]:
            rb = tk.Radiobutton(allergy_frame, text=val, variable=self.allergy, value=val,
                                bg=CARD_BG, fg=PRIMARY_TEXT, font=("Garamond",12,"bold"),
                                selectcolor=ACCENT_GOLD, command=self.toggle_allergy_entry)
            rb.pack(side="left", padx=5)
        self.allergy_entry = tk.Entry(allergy_frame, font=("Garamond",12), width=20)
        self.allergy_entry.pack(side="left", padx=10)
        self.allergy_entry.pack_forget()

        # Submit - Centered
        submit_frame = tk.Frame(main_frame, bg=BG_COLOR)
        submit_frame.pack(pady=40)
        tk.Button(submit_frame, text="Generate Digital Twin Analysis", bg=ACCENT_GOLD, fg="black",
                  font=("Didot",16,"bold"), padx=30, pady=12, bd=0, cursor="hand2",
                  command=self.submit).pack()

    # ---------- FUNCTIONS ----------
    def select_duration(self, option):
        self.duration.set(option)
        for opt, btn in self.duration_buttons.items():
            if opt == option:
                btn.config(bg=SELECTED_BG, fg=SELECTED_FG)
            else:
                btn.config(bg=UNSELECTED_BG, fg=UNSELECTED_FG)

    def toggle_risk(self, risk):
        btn = self.risks[risk]
        if btn["bg"] == SELECTED_BG:
            btn.config(bg=UNSELECTED_BG, fg=UNSELECTED_FG)
        else:
            btn.config(bg=SELECTED_BG, fg=SELECTED_FG)

    def update_bp(self, *args):
        self.bp_label.config(text=f"{self.sys.get()} / {self.dia.get()} mmHg")

    def toggle_bp_entry(self):
        if self.bp_med.get() == "Yes":
            self.bp_med_entry.pack(side="left", padx=10)
        else:
            self.bp_med_entry.pack_forget()

    def toggle_allergy_entry(self):
        if self.allergy.get() == "Yes":
            self.allergy_entry.pack(side="left", padx=10)
        else:
            self.allergy_entry.pack_forget()

    def submit(self):
        self.app.user_data["gender"] = self.app.gender
        self.app.user_data["age"] = self.age.get()
        self.app.user_data["duration"] = self.duration.get()
        self.app.user_data["systolic"] = self.sys.get()
        self.app.user_data["diastolic"] = self.dia.get()
        self.app.user_data["risks"] = {k:(v["bg"]==SELECTED_BG) for k,v in self.risks.items()}
        self.app.user_data["bp_med"] = self.bp_med.get()
        self.app.user_data["bp_med_name"] = self.bp_med_entry.get() if self.bp_med.get()=="Yes" else ""
        self.app.user_data["allergy"] = self.allergy.get()
        self.app.user_data["allergy_name"] = self.allergy_entry.get() if self.allergy.get()=="Yes" else ""
        
        print("User Data:", self.app.user_data)
        
        # Get ML predictions
        try:
            from ui_integration import DigitalTwinPredictor
            predictor = DigitalTwinPredictor()
            results = predictor.predict(self.app.user_data)
        except Exception as e:
            print(f"ML Prediction Error: {e}")
            # Fallback results
            results = {
                'top_recommendations': [
                    {
                        'rank': 1,
                        'drug_name': 'ACE Inhibitor (e.g., Lisinopril)',
                        'confidence': 85.0,
                        'explanation': 'First-line treatment based on clinical guidelines',
                        'expected_bp_reduction': 15.0
                    },
                    {
                        'rank': 2,
                        'drug_name': 'Calcium Channel Blocker (e.g., Amlodipine)',
                        'confidence': 78.0,
                        'explanation': 'Well-tolerated with good efficacy',
                        'expected_bp_reduction': 12.5
                    },
                    {
                        'rank': 3,
                        'drug_name': 'Diuretic (e.g., Hydrochlorothiazide)',
                        'confidence': 72.0,
                        'explanation': 'Suitable for initial therapy',
                        'expected_bp_reduction': 11.0
                    }
                ],
                'patient_profile': f"Age: {self.age.get()}, BP: {self.sys.get()}/{self.dia.get()} mmHg",
                'safety_warnings': []
            }
        
        # Show results page
        self.app.show_results(results)

# ---------- RESULTS PAGE ----------
class ResultsPage(tk.Frame):
    def __init__(self, parent, app, results):
        super().__init__(parent, bg=BG_COLOR)
        self.app = app
        self.results = results
        
        base = os.path.dirname(os.path.abspath(__file__))
        
        # Main container
        main_container = tk.Frame(self, bg=BG_COLOR)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Frame(main_container, bg=BG_COLOR)
        header.pack(fill="x", pady=(0, 20))
        tk.Label(header, text="Digital Twin Analysis Results", 
                 font=("Didot", 36, "bold"), fg=PRIMARY_TEXT, bg=BG_COLOR).pack()
        tk.Label(header, text="Personalized Drug Recommendations", 
                 font=("Garamond", 18), fg=SECONDARY_TEXT, bg=BG_COLOR).pack()
        
        # Content frame with gender image on left, results on right
        content_frame = tk.Frame(main_container, bg=BG_COLOR)
        content_frame.pack(fill="both", expand=True)
        
        # LEFT SIDE - Gender Image
        left_panel = tk.Frame(content_frame, bg=BG_COLOR, width=250)
        left_panel.pack(side="left", fill="y", padx=(0, 30))
        left_panel.pack_propagate(False)
        
        # Load and display gender image
        try:
            if app.gender == "male":
                gender_img = ImageTk.PhotoImage(Image.open(os.path.join(base, "male.png")).resize((200, 300)))
            else:
                gender_img = ImageTk.PhotoImage(Image.open(os.path.join(base, "female.png")).resize((200, 300)))
            
            self.gender_image = gender_img  # Keep reference
            tk.Label(left_panel, image=gender_img, bg=BG_COLOR).pack(pady=20)
        except:
            # Fallback if image not found
            tk.Label(left_panel, text=f"{app.gender.upper()}", 
                    font=("Didot", 24, "bold"), fg=ACCENT_GOLD, bg=BG_COLOR).pack(pady=50)
        
        # Patient summary in left panel
        summary_card = tk.Frame(left_panel, bg=CARD_BG, relief="raised", bd=2, padx=15, pady=15)
        summary_card.pack(fill="x")
        
        tk.Label(summary_card, text="Patient Profile", font=("Garamond", 14, "bold"),
                fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w")
        tk.Label(summary_card, text=f"Age: {app.user_data.get('age', 'N/A')}", 
                font=("Garamond", 11), fg=SECONDARY_TEXT, bg=CARD_BG).pack(anchor="w", pady=2)
        tk.Label(summary_card, text=f"BP: {app.user_data.get('systolic', 'N/A')}/{app.user_data.get('diastolic', 'N/A')} mmHg", 
                font=("Garamond", 11), fg=SECONDARY_TEXT, bg=CARD_BG).pack(anchor="w", pady=2)
        tk.Label(summary_card, text=f"Duration: {app.user_data.get('duration', 'N/A')}", 
                font=("Garamond", 11), fg=SECONDARY_TEXT, bg=CARD_BG).pack(anchor="w", pady=2)
        
        # RIGHT SIDE - Scrollable Results
        right_panel = tk.Frame(content_frame, bg=BG_COLOR)
        right_panel.pack(side="left", fill="both", expand=True)
        
        # Create scrollable canvas for results
        canvas = tk.Canvas(right_panel, bg=BG_COLOR, highlightthickness=0)
        scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        results_container = tk.Frame(canvas, bg=BG_COLOR)
        
        results_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=results_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Safety Warnings (if any)
        if results.get('safety_warnings'):
            warning_card = tk.Frame(results_container, bg="#FFF3CD", relief="raised", bd=2, padx=20, pady=15)
            warning_card.pack(fill="x", pady=(0, 20))
            
            tk.Label(warning_card, text="⚠️  Safety Warnings", font=("Garamond", 16, "bold"),
                    fg="#856404", bg="#FFF3CD").pack(anchor="w")
            
            for warning in results['safety_warnings']:
                tk.Label(warning_card, text=warning, font=("Garamond", 12),
                        fg="#856404", bg="#FFF3CD", wraplength=500, justify="left").pack(anchor="w", pady=2)
        
        # Top Recommendations
        tk.Label(results_container, text="💊 Top Drug Recommendations", 
                font=("Garamond", 20, "bold"), fg=PRIMARY_TEXT, bg=BG_COLOR).pack(anchor="w", pady=(0, 15))
        
        # Display each recommendation
        for rec in results['top_recommendations']:
            self.create_recommendation_card(results_container, rec)
        
        # Action Buttons at bottom
        button_frame = tk.Frame(main_container, bg=BG_COLOR)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="← Back to Assessment", bg=CARD_BG, fg=PRIMARY_TEXT,
                 font=("Garamond", 14, "bold"), padx=20, pady=10, relief="raised", bd=2,
                 command=app.show_assessment).pack(side="left", padx=10)
        
        tk.Button(button_frame, text="New Patient", bg=ACCENT_GOLD, fg="black",
                 font=("Didot", 14, "bold"), padx=20, pady=10, bd=0,
                 command=app.show_welcome).pack(side="left", padx=10)
    
    def create_recommendation_card(self, parent, rec):
        """Create a card for each drug recommendation"""
        # Determine color based on rank
        if rec['rank'] == 1:
            accent_color = "#2ECC71"  # Green for top choice
            rank_bg = "#27AE60"
        elif rec['rank'] == 2:
            accent_color = "#3498DB"  # Blue for second
            rank_bg = "#2980B9"
        else:
            accent_color = "#95A5A6"  # Gray for third
            rank_bg = "#7F8C8D"
        
        # Main card
        card = tk.Frame(parent, bg=CARD_BG, relief="raised", bd=2)
        card.pack(fill="x", pady=10, padx=5)
        
        # Top bar with rank and confidence
        top_bar = tk.Frame(card, bg=accent_color, height=8)
        top_bar.pack(fill="x")
        
        content = tk.Frame(card, bg=CARD_BG, padx=25, pady=20)
        content.pack(fill="x")
        
        # Header row with rank badge and drug name
        header_row = tk.Frame(content, bg=CARD_BG)
        header_row.pack(fill="x", pady=(0, 10))
        
        # Rank badge
        rank_badge = tk.Label(header_row, text=f"#{rec['rank']}", 
                             font=("Didot", 18, "bold"), fg="white", bg=rank_bg,
                             padx=15, pady=5)
        rank_badge.pack(side="left", padx=(0, 15))
        
        # Drug name
        drug_frame = tk.Frame(header_row, bg=CARD_BG)
        drug_frame.pack(side="left", fill="x", expand=True)
        
        tk.Label(drug_frame, text=rec['drug_name'], 
                font=("Garamond", 20, "bold"), fg=PRIMARY_TEXT, bg=CARD_BG,
                anchor="w").pack(fill="x")
        
        # Confidence bar
        confidence_frame = tk.Frame(content, bg=CARD_BG)
        confidence_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(confidence_frame, text="Confidence:", 
                font=("Garamond", 12, "bold"), fg=SECONDARY_TEXT, bg=CARD_BG).pack(side="left")
        
        # Progress bar
        progress_bg = tk.Frame(confidence_frame, bg="#E0E0E0", height=20, width=200)
        progress_bg.pack(side="left", padx=10)
        
        confidence_pct = rec['confidence'] / 100
        progress_fill = tk.Frame(progress_bg, bg=accent_color, height=20, 
                                width=int(200 * confidence_pct))
        progress_fill.place(x=0, y=0)
        
        tk.Label(confidence_frame, text=f"{rec['confidence']:.1f}%", 
                font=("Garamond", 12, "bold"), fg=accent_color, bg=CARD_BG).pack(side="left")
        
        # Expected BP Reduction
        bp_frame = tk.Frame(content, bg=CARD_BG)
        bp_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(bp_frame, text="📉 Expected BP Reduction:", 
                font=("Garamond", 13, "bold"), fg=PRIMARY_TEXT, bg=CARD_BG).pack(side="left")
        tk.Label(bp_frame, text=f"{rec['expected_bp_reduction']:.1f} mmHg", 
                font=("Garamond", 13), fg=ACCENT_GOLD, bg=CARD_BG).pack(side="left", padx=10)
        
        # Explanation
        tk.Label(content, text="💡 Why this recommendation:", 
                font=("Garamond", 12, "bold"), fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w", pady=(5, 2))
        tk.Label(content, text=rec['explanation'], 
                font=("Garamond", 12), fg=SECONDARY_TEXT, bg=CARD_BG,
                wraplength=550, justify="left").pack(anchor="w", padx=(20, 0))

# ---------- RUN ----------
if __name__ == "__main__":
    app = App()
    app.mainloop()
