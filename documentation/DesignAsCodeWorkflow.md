# Design-as-code Workflow

## Overview

This document describes our approach to creating architectural documentation using design-as-code principles. Rather than relying on traditional diagramming tools with graphical interfaces, we adopted a code-first approach using D2 (Declarative Diagramming), which allows us to version control our diagrams, generate them programmatically, and leverage AI assistance in their creation.

## Why design-as-code?

Traditional diagramming tools such as draw.io, Visio, and Lucidchart create sizable limitations for enterprise-scale documentation efforts. The primary challenge stems from their reliance on binary file formats, which makes version control extremely difficult as these files cannot be meaningfully compared or merged when conflicts arise. Furthermore, maintaining visual consistency across multiple diagrams becomes a manual and error-prone process, requiring constant vigilance to ensure styling and layout patterns stay uniform. These tools are also mostly proprietary also means that generating or updating diagrams programmatically is simply not feasible, limiting automation possibilities. Collaboration becomes problematic when multiple team members attempt to edit the same diagram simultaneously, often resulting in conflicts or lost work. And for this application most interestingly, AI assistants cannot easily understand or generate diagrams stored in these proprietary binary formats.

Treating diagrams as code removes these previously mentioned disadvantages. The text-based format of diagram code is Git-friendly, enabling proper version control with diffs that clearly show what has changed between revisions. Diagrams can be regenerated consistently from their source code, ensuring perfect reproducibility across different environments and time periods. Programmatic generation and updates become open the door to automation workflows that can keep documentation synchronized with code changes. AI collaboration emerges as a powerful capability, as AI assistants can read, understand, and generate diagram code just as they would any other source code. Finally, the inherent structure of code-based diagrams makes changes easier to track, review, and revert through standard development workflows.  

In addition, this git-friendly diagrams as code workflow integrates well with the existing documentation workflow. The images can be directly generated within the documentation folder structure, and thus be easily used within the documentation markdown files.

## The D2 Diagramming Language

### What is D2?

[D2](https://d2lang.com/) is a modern, declarative diagramming language created by Terrastruct. It compiles textual descriptions into visual diagrams, supporting various output formats (SVG, PNG, PDF).

### Key Features

The D2 system supports multiple diagram types including class diagrams, sequence diagrams, use cases, and flowcharts, providing comprehensive coverage for different documentation needs. Rich styling and theming capabilities are built into the language, with support for variables and extensive customization options that ensure visual consistency. Multiple layout algorithms, including ELK, dagre, and TALA, are available to handle different diagram types and complexity levels, automatically arranging components in visually pleasing configurations. An extensive library of shapes and icons encompasses common architectural elements such as person icons for actors, cylinders for databases, and ovals for use cases.

### Installation

```bash
# macOS
brew install d2

# Or download from https://github.com/terrastruct/d2/releases
```

### Basic Usage

```bash
# Compile D2 to SVG
d2 Architecture.d2 Architecture.svg

# Compile to PNG
d2 Architecture.d2 Architecture.png
```

## The design-as-code Workflow

### Phase 1: Textual Architecture Documentation

The first step in this workflow is to document the system architecture textually. This documentation should include a high-level architecture overview that details components, their responsibilities, and their interactions with one another. In case of this project we described the pipeline stages encompassing ingestion, MCP tooling, and LangGraph agents, providing a comprehensive view of how data moves through the system from initial input to final output. Technical details covering the models used, prompt engineering approaches, and tool boundaries were carefully documented to establish a complete technical reference. The agent orchestration section specifically detailed the four-node LangGraph workflow that forms the core of the intelligent processing system. This textual foundation served as the source of truth for all subsequent visual representations, ensuring that diagrams would accurately reflect the documented architecture rather than introducing inconsistencies.

### Phase 2: Creating D2 Templates by Hand

The second step is to manually create a first batch of D2 diagrams to establish visual patterns and conventions. This phase is critical for learning the D2 syntax to create effective diagrams. Through this hands-on process, a design of the system can be established by defining the style and visual hierarchy that would provide consistency across all future diagrams. Create reusable patterns for common elements such as containers, actors, and data flows that can be referenced in subsequent diagrams. Most importantly, this manual creation process produces concrete examples that AI assistants can learn from, serving as templates that demonstrates preferred styling and structural approaches.

### Phase 3: AI-Assisted Diagram Generation

With manual templates serving as references, an approach has been developped to leverage AI assistance for diagram creation and maintenance. This phase represents the culmination of the design-as-code strategy, where the patterns that were previously established manually become templates that AI systems can learn from and replicate.

The process of generating new diagrams begins with carefully structured prompts that should reference existing work. When requesting a new diagram, instruct the AI to create a D2 diagram showing the desired concept while explicitly directing it to use source code files as architectural reference material and existing D2 files as syntactic reference. It is neccessary to attach relevant files to these prompts, ensuring the AI understands not just what to create but how the actual system architecture is structured. It is good practice to specify which components and flows to include, providing clear requirements while leaving actual diagram creation to the AI's interpretation within the manually established architectural constraints.

Modifying existing diagrams follows a similar pattern but focuses on incremental changes rather than wholesale creation.

### Phase 4: Integration in documentation

#### Embedding in Documentation

```markdown
# An example of embedding the diagram into an architecture overview

![Architecture Diagram](./documentation/Architecture.png)

The system consists of three main subsystems:
- Ingestion: Parses PDFs and creates vector embeddings
- MCP Server: Exposes retrieval and web search tools
- Agent Orchestration: LangGraph workflow with Gemini models
```

## Best Practices

Our experience developing and maintaining design-as-code documentation has revealed several best practices that significantly improve outcomes and efficiency. These best practices are very similar to best practices within traditional design workflows.

Begin with generating simple diagrams rather than immediately tackling complex architectures proved essential for building competency and establishing patterns. The process started with a straightforward business flow of four nodes involving the business analyst actor. After this, an advancement was made into creating a use case diagram with clear boundaries, introducing container concepts and actor representations. The architecture diagram with its multiple containers and complex relationships came third, building on the patterns we had established in simpler contexts. Finally, a sequence diagram was created with its temporal ordering requirements. This staged approach ensured that the most common types of diagrams have been built, which the AI can use as a template for further iteration and generation of new diagrams.

Establishing variables early in the diagram development process creates a foundation for consistency and maintainability. By defining color schemes and reusable values at the top of each diagram file, a central reference point is created for all styling decisions. Similarly, embedding reusable configuration such as layout engine selection and theme identifiers in the variables block ensures that settings remain consistent and can be easily adjusted when needed.

Selecting appropriate layout engines for different diagram types significantly impacts the quality of generated visualizations. The dagre layout engine serves well for simple hierarchical layouts and functions as a sensible default for most basic diagrams. However, complex architectures could required more sophisticated layouts like the ELK layout engine, which handles diagrams with many crossing connections and complex relationship patterns more gracefully. The experimental TALA layout engine has not been considered as this is a paid engine.

Maintaining focus in individual diagrams by assigning each a single clear purpose prevents visual clutter and cognitive overload. This remains similarly true for the AI which has to analyse the template diagrams. Creating separate diagrams for different concerns and then link between them in existing documentation allows readers to navigate between related views while keeping each diagram conceptually clean.

Version control practices for diagram code mirror software development workflows. Both the source D2 files and the generated images are tracked in version control, recognizing that each serves a distinct purpose. The generated images help reviewers who may not have D2 installed locally understand proposed changes during code review, while the source D2 files remain the definitive source of truth that can be modified and regenerated. This dual tracking approach balances accessibility with maintainability.

## AI Collaboration

Effective collaboration with AI assistants for diagram generation requires providing comprehensive context and understanding the iterative nature of the process.

Providing adequate context begins with referencing existing patterns explicitly in the given prompts. Most importantly, attaching existing diagrams as context gives the AI actual examples to learn from rather than relying solely on verbal descriptions of preferences.

The iterative refinement process acknowledges that AI-generated diagrams typically require adjustment before reaching their final form. The workflow follows a cycle where an initial prompt leads to AI-generated D2 code, which is compiled and reviewed before refining the prompt and regenerating.

Learning from compiler errors creates a feedback loop that improves both the prompts and the understanding of D2's requirements. When the D2 compiler for instance reports that a referenced container does not exist, the prompts can be updated to include the error. This error-driven learning progressively refines the results of the prompts, encoding solutions to common problems so that future diagram generation encounters fewer issues. The compiler serves as a validator, catching mistakes that might otherwise propagate through the documentation.

## Results and Benefits

Before adopting design-as-code, our diagrams resided in draw.io as binary format files. Manual updates were required whenever the architecture changed, consuming significant time and introducing the risk of documentation drift. Styling inconsistencies accumulated across diagrams as different team members made modifications without reference to established patterns. Collaboration on diagram changes proved difficult, often requiring synchronous editing sessions or sequential handoffs to avoid conflicts. AI assistance remained entirely unavailable, as no automated system could meaningfully interact with our proprietary diagram formats.

All diagrams now reside in version control with meaningful diffs that clearly show what has changed between revisions, enabling reviewers to understand modifications without opening specialized tools The text-based format facilitates easy collaboration, as multiple team members can work on different diagrams simultaneously and resolve conflicts using standard merge tools. Lastly, AI systems can now generate and update diagrams based on our architectural documentation and existing diagram templates, dramatically accelerating diagram creation and maintenance.

## Resources

The following resources provide additional information and tools for implementing design-as-code workflows. The official D2 documentation at <https://d2lang.com/> offers comprehensive coverage of the language syntax, styling options, and advanced features. The D2 GitHub repository at <https://github.com/terrastruct/d2> provides access to the source code, issue tracking, and community discussions. The D2 Playground at <https://play.d2lang.com/> enables online experimentation with D2 syntax without requiring local installation, making it ideal for quick tests and learning exercises. Our own D2 templates reside in the documentation directory, providing concrete examples of our established patterns and serving as references for both human developers and AI assistants.
