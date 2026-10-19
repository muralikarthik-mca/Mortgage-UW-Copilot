"""Real browser + live Ollama checks. Requires requirements-dev.txt and Edge."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from dotenv import load_dotenv
from vector_store import backend_name

ROOT = Path(__file__).parent
load_dotenv(ROOT / '.env')


def main():
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:8502')
        launcher = page.get_by_role('button', name='Open AI underwriting assistant', exact=True)
        expect(launcher).to_be_visible(timeout=30000)
        heading = page.get_by_role('heading', name='Loan review workspace')
        before = heading.bounding_box()
        expect(page.get_by_role('dialog')).to_be_hidden()
        launcher.click()
        dialog = page.get_by_role('dialog')
        expect(dialog).to_be_visible()
        assert dialog.bounding_box()['width'] == 400
        assert heading.bounding_box() == before
        field = page.get_by_role('textbox', name='Ask about the selected loan')
        expect(field).to_be_focused()
        expect(page.get_by_role('button', name='What remains unverified about this income?', exact=True)).to_be_visible()
        field.fill('Draft retained when minimized')
        page.keyboard.press('Escape')
        expect(dialog).to_be_hidden()
        expect(launcher).to_be_focused()
        page.keyboard.press('Enter')
        expect(dialog).to_be_visible()
        expect(field).to_have_value('Draft retained when minimized')
        field.fill('Show me LN-10049 instead')
        page.get_by_role('button', name='Send question', exact=True).click()
        expect(dialog.get_by_text('This assistant can only review LN-10045.', exact=False)).to_be_visible(timeout=15000)
        question = 'Review the income documentation and outstanding items.'
        field.fill(question)
        page.keyboard.press('Enter')
        expect(dialog.get_by_role('status').filter(has_text='Retrieving evidence')).to_be_visible(timeout=10000)
        page.get_by_role('button', name='Close AI assistant', exact=True).click()
        expect(dialog).to_be_hidden()
        launcher.click()
        expect(dialog.get_by_text(question, exact=True)).to_be_visible()
        expect(dialog.get_by_text('Observed facts', exact=True)).to_be_visible(timeout=240000)
        expect(dialog.get_by_text('10 days before application.', exact=False)).to_be_visible()
        links = dialog.locator('a[href^="https://selling-guide.fanniemae.com/"]')
        expect(links.first).to_be_visible()
        assert links.first.get_attribute('target') == '_blank'
        assert 'noopener' in links.first.get_attribute('rel')
        page.screenshot(path=str(output / 'assistant-cited-desktop.png'))
        select = page.get_by_role('combobox', name='Select loan', exact=True)
        select.click()
        page.get_by_role('option', name='LN-10046', exact=True).click()
        expect(dialog).to_be_visible()
        expect(dialog.locator('.uw-loan')).to_have_text('LN-10046 · SALARIED')
        expect(dialog.locator('.uw-message.user .uw-body').filter(has_text=question)).to_have_count(0)
        expect(dialog.get_by_role('link')).to_have_count(0)
        select.click()
        page.get_by_role('option', name='LN-10045', exact=True).click()
        expect(dialog).to_be_visible()
        expect(dialog.get_by_text(question, exact=True)).to_be_visible()
        expect(dialog.get_by_text('Observed facts', exact=True)).to_be_visible()
        # Missing loan evidence is an explicit, deterministic insufficient state.
        select.click()
        page.get_by_role('option', name='LN-10049', exact=True).click()
        expect(dialog.locator('.uw-loan')).to_have_text('LN-10049 · SELF EMPLOYED')
        field.fill('How old is the paystub?')
        page.get_by_role('button', name='Send question', exact=True).click()
        expect(dialog.get_by_text('no dated paystub is recorded', exact=False)).to_be_visible(timeout=15000)
        expect(dialog.get_by_role('status').filter(has_text='Insufficient evidence')).to_be_visible()
        page.get_by_role('button', name='Close AI assistant', exact=True).click()
        # Persistent navigation works without a chat section appearing in the page.
        for name, title in [('Loans', 'Loan portfolio'), ('Guidelines', 'Guideline library'), ('RAG Evaluation', 'RAG evaluation')]:
            page.get_by_test_id('stSidebar').get_by_text(name, exact=True).click()
            expect(page.get_by_role('heading', name=title, exact=True).first).to_be_visible()
            if name == 'Guidelines':
                expected_backend = 'Pinecone' if backend_name() == 'pinecone' else 'Local NumPy'
                expect(page.get_by_text('Vector storage: ' + expected_backend, exact=True)).to_be_visible()
        mobile = browser.new_page(viewport={'width':390,'height':844}, is_mobile=True, has_touch=True)
        mobile.goto('http://127.0.0.1:8502')
        mobile.get_by_role('button', name='Open AI underwriting assistant', exact=True).click(timeout=30000)
        mobile_dialog = mobile.get_by_role('dialog')
        expect(mobile_dialog).to_be_visible()
        box = mobile_dialog.bounding_box()
        assert box['x'] >= 0 and box['width'] <= 390 and box['width'] >= 370
        assert box['height'] >= 800 and box['y'] >= 0
        assert mobile_dialog.get_attribute('aria-modal') == 'true'
        mobile_field = mobile.get_by_role('textbox', name='Ask about the selected loan')
        expect(mobile_field).to_be_focused()
        send = mobile.get_by_role('button', name='Send question', exact=True)
        send.focus()
        mobile.keyboard.press('Tab')
        expect(mobile.get_by_role('button', name='Close AI assistant', exact=True)).to_be_focused()
        mobile.keyboard.press('Shift+Tab')
        expect(send).to_be_focused()
        mobile.screenshot(path=str(output/'assistant-mobile.png'))
        mobile.keyboard.press('Escape')
        expect(mobile_dialog).to_be_hidden()
        assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert not errors, errors
        report = {'desktop_width':400, 'mobile_bounds':box, 'layout_unchanged':True,
                  'open_close_escape_focus':True, 'minimize_preserves_history':True,
                  'loan_context_isolation':True, 'live_submission_and_citations':True,
                  'loading_error_insufficient_states':True, 'navigation':True,
                  'vector_backend':backend_name(), 'browser_errors':errors}
        (output/'ui-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
        browser.close()


if __name__ == '__main__':
    main()
