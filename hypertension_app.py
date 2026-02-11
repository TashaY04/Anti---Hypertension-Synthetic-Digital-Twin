import tkinter as tk
from PIL import Image, ImageTk
import os

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
        self.male_img = ImageTk.PhotoImage(Image.open(os.path.join(base,"male.png")).resize((180,260)))
        self.female_img = ImageTk.PhotoImage(Image.open(os.path.join(base,"female.png")).resize((180,260)))

        card = tk.Frame(self, bg=CARD_BG, padx=30, pady=30, relief="raised", bd=2)
        card.pack(pady=20)

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

# ---------- ASSESSMENT PAGE ----------
class AssessmentPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_COLOR)
        self.app = app

        scroll_frame = ScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True)
        container = scroll_frame.scrollable_frame

        # Center frame
        main_frame = tk.Frame(container, bg=BG_COLOR)
        main_frame.pack(anchor="center", pady=30, fill="both", expand=True)

        # Header
        tk.Label(main_frame, text="Hypertension Clinical Profile",
                 font=("Didot",34,"bold"), fg=PRIMARY_TEXT, bg=BG_COLOR).pack(pady=(0,10))
        tk.Label(main_frame, text="Personalized inputs for digital twin simulation",
                 font=("Garamond",22,"bold"), fg=SECONDARY_TEXT, bg=BG_COLOR).pack(pady=(0,30))

        # ---------- CLINICAL SNAPSHOT ----------
        snapshot_card = tk.Frame(main_frame, bg=CARD_BG, padx=50, pady=50, relief="raised", bd=2)
        snapshot_card.pack(pady=20, fill="x", expand=True)

        tk.Label(snapshot_card, text="Age", font=("Garamond",18,"bold"), fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w")
        self.age = tk.IntVar(value=30)
        tk.Scale(snapshot_card, from_=18, to=90, orient="horizontal", variable=self.age,
                 bg=CARD_BG, highlightthickness=0).pack(fill="x", pady=(0,15))

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
        self.sys = tk.IntVar(value=140)
        self.dia = tk.IntVar(value=90)
        tk.Scale(bp_frame, from_=90, to=200, variable=self.sys, orient="vertical",
                 label="Systolic", bg=CARD_BG, highlightthickness=0).pack(side="left", expand=True, fill="y", padx=40)
        self.bp_label = tk.Label(bp_frame, text=f"{self.sys.get()} / {self.dia.get()} mmHg",
                                 font=("Garamond",16,"bold"), fg=ACCENT_GOLD, bg=CARD_BG)
        self.bp_label.pack(side="left", padx=20)
        tk.Scale(bp_frame, from_=60, to=120, variable=self.dia, orient="vertical",
                 label="Diastolic", bg=CARD_BG, highlightthickness=0).pack(side="left", expand=True, fill="y", padx=40)
        self.sys.trace_add("write", self.update_bp)
        self.dia.trace_add("write", self.update_bp)

        # ---------- RISK FACTORS ----------
        risk_card = tk.Frame(main_frame, bg=CARD_BG, padx=50, pady=50, relief="raised", bd=2)
        risk_card.pack(pady=20, fill="x", expand=True)
        tk.Label(risk_card, text="Risk Factors", font=("Garamond",18,"bold"), fg=PRIMARY_TEXT, bg=CARD_BG).pack(anchor="w", pady=(0,10))
        risk_frame = tk.Frame(risk_card, bg=CARD_BG)
        risk_frame.pack(anchor="w", fill="x")

        if self.app.gender == "male":
            risk_list = ["Diabetes","Kidney Disease","Smoker","High Stress","Sedentary Lifestyle","High Cholesterol"]
        else:
            risk_list = ["Diabetes","Kidney Disease","High Stress","Sedentary Lifestyle","Hormonal Imbalance","Pregnancy / Postpartum"]

        self.risks = {}
        for risk in risk_list:
            btn = tk.Button(risk_frame, text=risk, bg=UNSELECTED_BG, fg=UNSELECTED_FG,
                            font=("Garamond",14,"bold"), relief="ridge", bd=2,
                            padx=12, pady=6, command=lambda r=risk: self.toggle_risk(r))
            btn.pack(side="left", padx=6, pady=6)
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
        self.bp_med_entry = tk.Entry(bp_frame, font=("Garamond",12))
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
        self.allergy_entry = tk.Entry(allergy_frame, font=("Garamond",12))
        self.allergy_entry.pack(side="left", padx=10)
        self.allergy_entry.pack_forget()

        # Submit
        tk.Button(main_frame, text="Generate Digital Twin Analysis", bg=ACCENT_GOLD, fg="black",
                  font=("Didot",16,"bold"), padx=30, pady=12, bd=0, cursor="hand2",
                  command=self.submit).pack(pady=40)

    # ---------- FUNCTIONS ----------
    def select_duration(self, option):
        # Single-select duration
        self.duration.set(option)
        for opt, btn in self.duration_buttons.items():
            if opt == option:
                btn.config(bg=SELECTED_BG, fg=SELECTED_FG)
            else:
                btn.config(bg=UNSELECTED_BG, fg=UNSELECTED_FG)

    def toggle_risk(self, risk):
        # Multi-select risk
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
        print(self.app.user_data)

# ---------- RUN ----------
if __name__ == "__main__":
    app = App()
    app.mainloop()
