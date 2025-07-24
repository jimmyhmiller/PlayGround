import Cocoa

class MainViewController: NSViewController {
    private var codeBlocks: [CodeBlockView] = []
    private var scoresPanel: ScoresPanelView!
    private var scrollView: NSScrollView!
    private var contentView: NSView!
    
    override func loadView() {
        view = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 800))
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor(red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0).cgColor
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupSimpleCodeBlock()
    }
    
    private func setupSimpleCodeBlock() {
        let routeBlock = CodeBlockView(code: """
@app.route("/scores")
def score():
    track()
    scs = scores.getScores()
    return render_template("scores.html", scs=scs)
""", language: .python)
        routeBlock.frame = NSRect(x: 50, y: 600, width: 600, height: 120)
        view.addSubview(routeBlock)
        
        let trackBlock = CodeBlockView(code: """
def track():
    \"\"\"Track when people access the scores page\"\"\"
    global visits
    visits += 1
    app.logger.info("Scores hit: " + str(visits))
""", language: .python)
        trackBlock.frame = NSRect(x: 50, y: 460, width: 600, height: 120)
        view.addSubview(trackBlock)
        
        let getScoresBlock = CodeBlockView(code: """
def getScores():
    return sorted(scores, key=itemgetter("value"), reverse=True)
""", language: .python)
        getScoresBlock.frame = NSRect(x: 50, y: 380, width: 600, height: 60)
        view.addSubview(getScoresBlock)
        
        let htmlBlock = CodeBlockView(code: """
{% extends "layout.html" %}
{% block body %}
<h2>Scores</h2>
<ul>
    {% for score in scs %}
    <li>{{score.name}} - {{score.value}}</li>
    {% endfor %}
</ul>
{% endblock %}
""", language: .html)
        htmlBlock.frame = NSRect(x: 50, y: 220, width: 600, height: 140)
        view.addSubview(htmlBlock)
        
        // Add web view
        let webPanel = ScoresPanelView()
        webPanel.frame = NSRect(x: 770, y: 350, width: 300, height: 400)
        view.addSubview(webPanel)
    }
    
    private func setupUI() {
        setupScrollView()
        setupScoresPanel()
        setupCodeBlocks()
    }
    
    private func setupScrollView() {
        scrollView = NSScrollView()
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.borderType = .noBorder
        scrollView.backgroundColor = .clear
        
        contentView = NSView()
        contentView.wantsLayer = true
        contentView.layer?.backgroundColor = NSColor(red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0).cgColor
        
        scrollView.documentView = contentView
        view.addSubview(scrollView)
        
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: view.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
    
    private func setupScoresPanel() {
        scoresPanel = ScoresPanelView()
        scoresPanel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(scoresPanel)
        
        NSLayoutConstraint.activate([
            scoresPanel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 20),
            scoresPanel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            scoresPanel.widthAnchor.constraint(equalToConstant: 300),
            scoresPanel.heightAnchor.constraint(equalToConstant: 400)
        ])
    }
    
    private func setupCodeBlocks() {
        let routeBlock = CodeBlockView(code: """
@app.route("/scores")
def score():
    track()
    scs = scores.getScores()
    return render_template("scores.html", scs=scs)
""", language: .python)
        
        let trackBlock = CodeBlockView(code: """
def track():
    \"\"\"Track when people access the scores page\"\"\"
    global visits
    visits += 1
    app.logger.info("Scores hit: " + str(visits))
""", language: .python)
        
        let getScoresBlock = CodeBlockView(code: """
def getScores():
    return sorted(scores, key=itemgetter("value"), reverse=True)
""", language: .python)
        
        let htmlBlock = CodeBlockView(code: """
{% extends "layout.html" %}
{% block body %}
<h2>Scores</h2>
<ul>
    {% for score in scs %}
    <li>{{score.name}} - {{score.value}}</li>
    {% endfor %}
</ul>
{% endblock %}
""", language: .html)
        
        let htmlDocBlock = CodeBlockView(code: """
<!doctype html>
<html>
<head>
    <title>Chris</title>
    <link rel="stylesheet" type="text/css" href="{{url_for('static',
</head>
<body>
    <div class="page">
        {% block body %}{% endblock %}
    </div>
</body>
</html>
""", language: .html)
        
        codeBlocks = [routeBlock, trackBlock, getScoresBlock, htmlBlock, htmlDocBlock]
        
        // Set up auto-layout for code blocks flowing downwards
        var lastBottomAnchor: NSLayoutYAxisAnchor = contentView.topAnchor
        var totalHeight: CGFloat = 50
        
        for (index, block) in codeBlocks.enumerated() {
            block.translatesAutoresizingMaskIntoConstraints = false
            contentView.addSubview(block)
            
            NSLayoutConstraint.activate([
                block.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 50),
                block.widthAnchor.constraint(equalToConstant: 600)
            ])
            
            if index == 0 {
                block.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 50).isActive = true
            } else {
                block.topAnchor.constraint(equalTo: lastBottomAnchor, constant: 20).isActive = true
                totalHeight += 20
            }
            
            // Set dynamic height based on content
            let lines = block.code.components(separatedBy: .newlines).count
            let height = max(60, CGFloat(lines * 20 + 40))
            block.heightAnchor.constraint(equalToConstant: height).isActive = true
            
            lastBottomAnchor = block.bottomAnchor
            totalHeight += height
        }
        
        // Set content view height to accommodate all blocks
        totalHeight += 50 // bottom padding
        contentView.heightAnchor.constraint(equalToConstant: totalHeight).isActive = true
        contentView.widthAnchor.constraint(equalTo: scrollView.widthAnchor).isActive = true
    }
}