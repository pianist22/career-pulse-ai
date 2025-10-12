"""
Skill Matching and Extraction Module
Extracts technical and domain-specific skills from resume text
"""

import re
from typing import List, Set, Dict
from pathlib import Path


class SkillMatcher:
    """
    Matches and extracts skills from resume text using predefined skill lists and patterns
    """
    
    def __init__(self):
        """Initialize skill matcher with predefined skill categories"""
        self.skills_db = self._load_skills_database()
        
    def _load_skills_database(self) -> Dict[str, List[str]]:
        """
        Load comprehensive skills database
        
        Returns:
            Dictionary with skill categories and their lists
        """
        return {
            "programming_languages": [
                "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "PHP", "Ruby", "Go", "Rust",
                "Swift", "Kotlin", "Scala", "R", "MATLAB", "Perl", "Shell", "Bash", "PowerShell",
                "Assembly", "Fortran", "COBOL", "Ada", "Lisp", "Prolog", "Haskell", "Erlang", "Clojure",
                "Julia", "Dart", "Lua", "VBA", "Objective-C", "Delphi", "Pascal", "Visual Basic"
            ],
            
            "web_technologies": [
                "HTML", "CSS", "JavaScript", "TypeScript", "React", "Angular", "Vue.js", "Vue",
                "Node.js", "Express.js", "Next.js", "Nuxt.js", "Svelte", "jQuery", "Bootstrap",
                "Tailwind CSS", "SASS", "SCSS", "Less", "Webpack", "Babel", "Gulp", "Grunt",
                "Redux", "Vuex", "MobX", "GraphQL", "Apollo", "REST", "RESTful", "API", "JSON", "XML"
            ],
            
            "databases": [
                "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Oracle", "SQL Server",
                "MariaDB", "Cassandra", "Neo4j", "Elasticsearch", "DynamoDB", "CouchDB",
                "InfluxDB", "TimescaleDB", "ClickHouse", "BigQuery", "Snowflake", "Redshift",
                "SQL", "NoSQL", "T-SQL", "PL/SQL", "MongoDB", "Firebase", "Supabase"
            ],
            
            "cloud_platforms": [
                "AWS", "Amazon Web Services", "Azure", "Microsoft Azure", "GCP", "Google Cloud Platform",
                "Google Cloud", "IBM Cloud", "Oracle Cloud", "DigitalOcean", "Linode", "Vultr",
                "Heroku", "Netlify", "Vercel", "Cloudflare", "Alibaba Cloud", "Tencent Cloud"
            ],
            
            "devops_tools": [
                "Docker", "Kubernetes", "Jenkins", "GitLab CI", "GitHub Actions", "Travis CI",
                "CircleCI", "Azure DevOps", "TeamCity", "Bamboo", "Ansible", "Terraform",
                "Chef", "Puppet", "SaltStack", "Vagrant", "Packer", "Helm", "Istio",
                "Prometheus", "Grafana", "ELK Stack", "Splunk", "Nagios", "Zabbix"
            ],
            
            "ml_ai_tools": [
                "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy", "SciPy",
                "Matplotlib", "Seaborn", "Plotly", "OpenCV", "NLTK", "spaCy", "Gensim",
                "Transformers", "Hugging Face", "Jupyter", "Colab", "MLflow", "Kubeflow",
                "Apache Spark", "Hadoop", "Hive", "Pig", "Mahout", "Weka", "RapidMiner"
            ],
            
            "frameworks": [
                "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "Laravel", "Rails",
                "Express.js", "NestJS", "ASP.NET", "ASP.NET Core", "Symfony", "CodeIgniter",
                "Yii", "Zend", "CakePHP", "Slim", "Sinatra", "Phoenix", "Gin", "Echo",
                "Fiber", "Koa", "Hapi", "Sails.js", "Meteor", "LoopBack", "Strapi"
            ],
            
            "mobile_development": [
                "React Native", "Flutter", "Ionic", "Xamarin", "Cordova", "PhoneGap",
                "Android Studio", "Xcode", "Swift", "Kotlin", "Java", "Objective-C",
                "Dart", "Ionic", "NativeScript", "Appcelerator", "Titanium"
            ],
            
            "testing_tools": [
                "Jest", "Mocha", "Chai", "Jasmine", "Cypress", "Selenium", "Playwright",
                "Puppeteer", "JUnit", "TestNG", "Mockito", "PowerMock", "Cucumber",
                "SpecFlow", "Behave", "Pytest", "Unittest", "Nose", "Robot Framework"
            ],
            
            "version_control": [
                "Git", "GitHub", "GitLab", "Bitbucket", "SVN", "Mercurial", "Perforce",
                "TFS", "Azure DevOps", "SourceTree", "TortoiseGit", "SmartGit"
            ],
            
            "project_management": [
                "Agile", "Scrum", "Kanban", "Waterfall", "JIRA", "Confluence", "Trello",
                "Asana", "Monday.com", "Notion", "Slack", "Microsoft Teams", "Zoom",
                "PMP", "PRINCE2", "ITIL", "Six Sigma", "Lean"
            ],
            
            "operating_systems": [
                "Linux", "Ubuntu", "CentOS", "Red Hat", "Debian", "Fedora", "Windows",
                "Windows Server", "macOS", "Unix", "FreeBSD", "OpenBSD", "Solaris"
            ],
            
            "networking": [
                "TCP/IP", "HTTP", "HTTPS", "FTP", "SSH", "DNS", "DHCP", "VPN", "LAN",
                "WAN", "Firewall", "Load Balancer", "CDN", "SSL", "TLS", "IPv4", "IPv6"
            ],
            
            "security": [
                "OAuth", "JWT", "SAML", "LDAP", "Active Directory", "Kerberos", "SSL",
                "TLS", "Encryption", "Penetration Testing", "Vulnerability Assessment",
                "SIEM", "SOC", "CISSP", "CISA", "CISM", "Security+", "CEH"
            ]
        }
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract all skills from resume text
        
        Args:
            text: Resume text to analyze
            
        Returns:
            List of found skills
        """
        if not text:
            return []
        
        found_skills = set()
        text_lower = text.lower()
        
        # Extract skills from each category
        for category, skills in self.skills_db.items():
            for skill in skills:
                skill_lower = skill.lower()
                
                # Exact match
                if skill_lower in text_lower:
                    found_skills.add(skill)
                
                # Pattern-based matching for variations
                patterns = [
                    rf'\b{re.escape(skill_lower)}\b',  # Word boundary match
                    rf'\b{re.escape(skill_lower)}\s*\+',  # Skill with plus
                    rf'\b{re.escape(skill_lower)}\s*/\s*',  # Skill with slash
                    rf'\b{re.escape(skill_lower)}\s*&\s*',  # Skill with ampersand
                ]
                
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        found_skills.add(skill)
        
        # Additional pattern-based skill extraction
        additional_skills = self._extract_pattern_based_skills(text)
        found_skills.update(additional_skills)
        
        return sorted(list(found_skills))
    
    def _extract_pattern_based_skills(self, text: str) -> Set[str]:
        """
        Extract skills using pattern matching for common skill mentions
        
        Args:
            text: Resume text
            
        Returns:
            Set of additional skills found
        """
        skills = set()
        
        # Skills with "experience" or "proficient"
        exp_patterns = [
            r'(?:experience|proficient|skilled|expert|familiar|knowledgeable)\s+(?:in|with|of)\s+([A-Za-z\s+&.,-]+?)(?:\s|,|\.|$)',
            r'(?:worked\s+with|used|utilized|implemented)\s+([A-Za-z\s+&.,-]+?)(?:\s|,|\.|$)',
            r'(?:technologies?|tools?|languages?|frameworks?|platforms?)\s*:?\s*([A-Za-z\s+&.,-]+?)(?:\n|$)',
        ]
        
        for pattern in exp_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                skill_text = match.group(1).strip()
                # Split and clean potential skills
                potential_skills = re.split(r'[,;&\n]', skill_text)
                for skill in potential_skills:
                    skill = skill.strip()
                    if len(skill) > 2 and skill.replace(' ', '').isalnum():
                        skills.add(skill.title())
        
        return skills
    
    def get_skill_categories(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize found skills by their type
        
        Args:
            skills: List of extracted skills
            
        Returns:
            Dictionary mapping categories to skills
        """
        categorized = {}
        
        for category, skill_list in self.skills_db.items():
            category_skills = []
            for skill in skills:
                if any(skill.lower() == s.lower() or skill.lower() in s.lower() or s.lower() in skill.lower() 
                       for s in skill_list):
                    category_skills.append(skill)
            
            if category_skills:
                categorized[category] = category_skills
        
        return categorized
    
    def calculate_skill_score(self, text: str) -> Dict[str, int]:
        """
        Calculate skill density scores for different categories
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with skill category scores
        """
        scores = {}
        
        for category, skills in self.skills_db.items():
            count = 0
            for skill in skills:
                if skill.lower() in text.lower():
                    count += 1
            scores[category] = count
        
        return scores
